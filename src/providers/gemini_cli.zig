const std = @import("std");
const builtin = @import("builtin");
const root = @import("root.zig");
const platform = @import("../platform.zig");
const json_util = @import("../json_util.zig");

const Provider = root.Provider;
const ChatRequest = root.ChatRequest;
const ChatResponse = root.ChatResponse;
const ChatMessage = root.ChatMessage;

const log = std.log.scoped(.gemini_cli);

/// Provider that delegates to the `gemini` CLI (Google Gemini).
///
/// Implements a Lazy Persistent Singleton:
/// 1. Spawns `gemini --experimental-acp` on first use.
/// 2. Performs a JSON-RPC 2.0 handshake to create a session.
/// 3. Keeps the process alive for subsequent requests.
/// 4. Communicates via JSON-RPC 2.0 over stdio.
pub const GeminiCliProvider = struct {
    allocator: std.mem.Allocator,
    model: []const u8,
    
    /// Persistent state
    child: ?std.process.Child = null,
    mutex: std.Thread.Mutex = .{},
    session_id: ?[]const u8 = null,
    next_id: u32 = 1,
    read_buffer: std.ArrayListUnmanaged(u8) = .empty,

    const DEFAULT_MODEL = "gemini-2.5-pro";
    const CLI_NAME = "gemini";

    pub fn init(allocator: std.mem.Allocator, model: ?[]const u8) !GeminiCliProvider {
        // Just verify CLI is in PATH, don't start it yet.
        try checkCliVersion(allocator, CLI_NAME);
        return .{
            .allocator = allocator,
            .model = model orelse DEFAULT_MODEL,
        };
    }

    /// Create a Provider vtable interface.
    pub fn provider(self: *GeminiCliProvider) Provider {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }

    const vtable = Provider.VTable{
        .chatWithSystem = chatWithSystemImpl,
        .chat = chatImpl,
        .supportsNativeTools = supportsNativeToolsImpl,
        .getName = getNameImpl,
        .deinit = deinitImpl,
        .supports_vision = supportsVisionImpl,
    };

    fn chatWithSystemImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        system_prompt: ?[]const u8,
        message: []const u8,
        model: []const u8,
        _: f64,
    ) anyerror![]const u8 {
        const self: *GeminiCliProvider = @ptrCast(@alignCast(ptr));
        const effective_model = if (model.len > 0) model else self.model;

        // Combine system prompt with message
        const prompt = if (system_prompt) |sys|
            try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ sys, message })
        else
            try allocator.dupe(u8, message);
        defer allocator.free(prompt);

        return self.sendPrompt(allocator, prompt, effective_model);
    }

    fn chatImpl(
        ptr: *anyopaque,
        allocator: std.mem.Allocator,
        request: ChatRequest,
        model: []const u8,
        _: f64,
    ) anyerror!ChatResponse {
        const self: *GeminiCliProvider = @ptrCast(@alignCast(ptr));
        const effective_model = if (model.len > 0) model else self.model;

        const prompt = extractLastUserMessage(request.messages) orelse return error.NoUserMessage;
        const content = try self.sendPrompt(allocator, prompt, effective_model);
        return ChatResponse{ .content = content, .model = try allocator.dupe(u8, effective_model) };
    }

    fn supportsNativeToolsImpl(_: *anyopaque) bool {
        return false;
    }

    fn supportsVisionImpl(_: *anyopaque) bool {
        return false;
    }

    fn getNameImpl(_: *anyopaque) []const u8 {
        return "gemini-cli";
    }

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *GeminiCliProvider = @ptrCast(@alignCast(ptr));
        self.mutex.lock();
        defer self.mutex.unlock();
        self.stopInternal();
        self.read_buffer.deinit(self.allocator);
    }

    fn stopInternal(self: *GeminiCliProvider) void {
        if (self.child) |*child| {
            if (child.stdin) |stdin| stdin.close();
            _ = child.kill() catch {};
            _ = child.wait() catch {};
            self.child = null;
        }
        if (self.session_id) |sid| {
            self.allocator.free(sid);
            self.session_id = null;
        }
        self.read_buffer.clearRetainingCapacity();
    }

    /// Ensure the gemini agent process is running and a session is created.
    fn ensureStarted(self: *GeminiCliProvider) !void {
        if (self.child != null and self.session_id != null) return;

        // Clean up any stale state
        self.stopInternal();

        const argv = [_][]const u8{ CLI_NAME, "--experimental-acp", "--approval-mode", "yolo" };
        var child = std.process.Child.init(&argv, self.allocator);
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Ignore; // Ignore stderr to avoid blocking

        try child.spawn();
        self.child = child;

        // Step 2: session/new
        const cwd = try std.process.getCwdAlloc(self.allocator);
        defer self.allocator.free(cwd);

        const id = self.next_id;
        self.next_id += 1;

        var req_buf: std.ArrayListUnmanaged(u8) = .empty;
        defer req_buf.deinit(self.allocator);
        var writer = req_buf.writer(self.allocator);
        try writer.print("{{\"jsonrpc\":\"2.0\",\"id\":{d},\"method\":\"session/new\",\"params\":{{\"cwd\":", .{id});
        try json_util.appendJsonString(&req_buf, self.allocator, cwd);
        try writer.writeAll(",\"mcpServers\":[]}}}}\n");

        try self.child.?.stdin.?.writeAll(req_buf.items);

        // Read responses until we get the result for our ID
        while (true) {
            const line = try self.readLine(self.allocator);
            defer self.allocator.free(line);

            const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, line, .{}) catch continue;
            defer parsed.deinit();

            if (parsed.value != .object) continue;
            const obj = parsed.value.object;

            if (obj.get("id")) |resp_id| {
                if (resp_id == .integer and resp_id.integer == id) {
                    const result = obj.get("result") orelse {
                        if (obj.get("error")) |err| {
                            log.err("Gemini ACP session/new failed: {any}", .{err});
                        }
                        return error.InvalidHandshake;
                    };
                    if (result != .object) return error.InvalidHandshake;
                    const sid = result.object.get("sessionId") orelse return error.InvalidHandshake;
                    if (sid != .string) return error.InvalidHandshake;

                    self.session_id = try self.allocator.dupe(u8, sid.string);
                    break;
                }
            }
        }
    }

    fn sendPrompt(self: *GeminiCliProvider, allocator: std.mem.Allocator, prompt: []const u8, model: []const u8) ![]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.ensureStarted();

        const id = self.next_id;
        self.next_id += 1;

        // session/prompt: {"jsonrpc":"2.0","id":1,"method":"session/prompt","params":{"sessionId":"...","prompt":[{"type":"text","text":"..."}]}}
        var msg_buf: std.ArrayListUnmanaged(u8) = .empty;
        defer msg_buf.deinit(allocator);
        try msg_buf.appendSlice(allocator, "{\"jsonrpc\":\"2.0\",\"id\":");
        try std.fmt.format(msg_buf.writer(allocator), "{d}", .{id});
        try msg_buf.appendSlice(allocator, ",\"method\":\"session/prompt\",\"params\":{\"sessionId\":");
        try json_util.appendJsonString(&msg_buf, allocator, self.session_id.?);
        try msg_buf.appendSlice(allocator, ",\"model\":");
        try json_util.appendJsonString(&msg_buf, allocator, model);
        try msg_buf.appendSlice(allocator, ",\"prompt\":[{\"type\":\"text\",\"text\":");
        
        // Escape prompt using our local json_util
        try json_util.appendJsonString(&msg_buf, allocator, prompt);
        
        try msg_buf.appendSlice(allocator, "}]}}\n");

        try self.child.?.stdin.?.writeAll(msg_buf.items);

        // Accumulate output
        var result_buf: std.ArrayListUnmanaged(u8) = .empty;
        defer result_buf.deinit(allocator);

        while (true) {
            const line = self.readLine(allocator) catch |err| {
                self.stopInternal(); // Restart on next call
                return err;
            };
            defer allocator.free(line);

            const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch continue;
            defer parsed.deinit();

            if (parsed.value != .object) continue;
            const obj = parsed.value.object;

            // session/update notifications
            if (obj.get("method")) |method| {
                if (method == .string and std.mem.eql(u8, method.string, "session/update")) {
                    const params = (obj.get("params") orelse continue).object;
                    const update = (params.get("update") orelse continue).object;
                    const content = (update.get("content") orelse continue).object;
                    if (content.get("text")) |text_val| {
                        if (text_val == .string) {
                            try result_buf.appendSlice(allocator, text_val.string);
                        }
                    }
                    continue;
                }
            }

            // Final result response
            if (obj.get("id")) |resp_id| {
                const id_match = switch (resp_id) {
                    .integer => |vid| vid == id,
                    .float => |vid| @as(u32, @intFromFloat(vid)) == id,
                    else => false,
                };
                if (id_match) {
                    // This is our response
                    if (obj.get("result")) |res| {
                        if (res == .object) {
                             if (res.object.get("content")) |c| {
                                 if (c == .string) {
                                     if (c.string.len > 0) {
                                         result_buf.clearRetainingCapacity();
                                         try result_buf.appendSlice(allocator, c.string);
                                     }
                                 }
                             }
                        }
                    }
                    break;
                }
            }
        }

        if (result_buf.items.len == 0) return error.NoResultInOutput;
        return try result_buf.toOwnedSlice(allocator);
    }

    fn readLine(self: *GeminiCliProvider, allocator: std.mem.Allocator) ![]const u8 {
        const stdout = self.child.?.stdout.?;

        while (true) {
            // Check if we already have a full line in the buffer
            if (std.mem.indexOfScalar(u8, self.read_buffer.items, '\n')) |pos| {
                const line = try allocator.dupe(u8, self.read_buffer.items[0..pos]);
                errdefer allocator.free(line);
                
                // Remove the line (including \n) from the persistent buffer
                const total_len = pos + 1;
                std.mem.copyForwards(u8, self.read_buffer.items[0 .. self.read_buffer.items.len - total_len], self.read_buffer.items[total_len..]);
                self.read_buffer.items.len -= total_len;

                // Strip non-JSON prefixes
                const trimmed = std.mem.trim(u8, line, " \t\r");
                if (trimmed.len > 0 and (trimmed[0] == '{' or trimmed[0] == '[')) {
                    return line;
                }
                allocator.free(line);
                continue;
            }

            // Read more data from stdout
            var buf: [4096]u8 = undefined;
            const amt = try stdout.read(&buf);
            if (amt == 0) return error.EndOfStream;
            try self.read_buffer.appendSlice(self.allocator, buf[0..amt]);
        }
    }

    /// Fetch available model names by running `gemini -p \"/model\" -o stream-json`.
    pub fn fetchModels(allocator: std.mem.Allocator) [][]const u8 {
        if (builtin.is_test) return &.{};
        return fetchModelsInternal(allocator) catch &.{};
    }

    fn fetchModelsInternal(allocator: std.mem.Allocator) ![][]const u8 {
        var argv_storage: [10][]const u8 = undefined;
        var argv_list = std.ArrayList([]const u8).fromFixedBuffer(&argv_storage);
        try argv_list.append(CLI_NAME);
        try argv_list.append("-p");
        try argv_list.append("/model");
        try argv_list.append("-o");
        try argv_list.append("stream-json");
        const argv = argv_list.items;

        var child = std.process.Child.init(argv, allocator);
        child.stdin_behavior = .Close;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Ignore;

        try child.spawn();

        const max_output: usize = 64 * 1024;
        const out = child.stdout.?.readToEndAlloc(allocator, max_output) catch |err| {
            _ = child.wait() catch {};
            return err;
        };
        defer allocator.free(out);

        const term = try child.wait();
        switch (term) {
            .Exited => |code| {
                if (code != 0) return error.CliProcessFailed;
            },
            else => return error.CliProcessFailed,
        }

        // Extract tokens that look like Gemini model IDs ("gemini-*").
        var models: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer {
            for (models.items) |m| allocator.free(m);
            models.deinit(allocator);
        }

        var line_iter = std.mem.splitScalar(u8, out, '\n');
        while (line_iter.next()) |line| {
            if (line.len == 0) continue;
            const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch continue;
            defer parsed.deinit();
            if (parsed.value != .object) continue;
            const obj = parsed.value.object;
            const type_val = obj.get("type") orelse continue;
            if (type_val != .string) continue;
            
            var text: ?[]const u8 = null;
            if (std.mem.eql(u8, type_val.string, "message") or std.mem.eql(u8, type_val.string, "output")) {
                 if (obj.get("content")) |c| if (c == .string) { text = c.string; };
            }

            if (text) |t| {
                var token_iter = std.mem.tokenizeScalar(u8, t, ' ');
                while (token_iter.next()) |tok| {
                    const clean = std.mem.trim(u8, tok, " \t\r,;:");
                    if (clean.len > "gemini-".len and std.mem.startsWith(u8, clean, "gemini-")) {
                        try models.append(allocator, try allocator.dupe(u8, clean));
                    }
                }
            }
        }

        return models.toOwnedSlice(allocator);
    }
};

/// Run `<cli> --version` and verify exit code 0.
fn checkCliVersion(allocator: std.mem.Allocator, cli_name: []const u8) !void {
    const argv = [_][]const u8{ cli_name, "--version" };
    var child = std.process.Child.init(&argv, allocator);
    child.stdin_behavior = .Close;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Ignore;
    child.spawn() catch return error.CliNotFound;
    const out = child.stdout.?.readToEndAlloc(allocator, 4096) catch {
        _ = child.wait() catch {};
        return error.CliNotFound;
    };
    allocator.free(out);
    const term = child.wait() catch return error.CliNotFound;
    if (term == .Exited and term.Exited == 0) return;
    return error.CliNotFound;
}

fn extractLastUserMessage(messages: []const ChatMessage) ?[]const u8 {
    var i = messages.len;
    while (i > 0) {
        i -= 1;
        if (messages[i].role == .user) return messages[i].content;
    }
    return null;
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

test "GeminiCliProvider.getNameImpl returns gemini-cli" {
    const vt = GeminiCliProvider.vtable;
    var dummy: u8 = 0;
    try std.testing.expectEqualStrings("gemini-cli", vt.getName(@ptrCast(&dummy)));
}

test "GeminiCliProvider vtable has correct function pointers" {
    const vt = GeminiCliProvider.vtable;
    var dummy: u8 = 0;
    try std.testing.expectEqualStrings("gemini-cli", vt.getName(@ptrCast(&dummy)));
    try std.testing.expect(!vt.supportsNativeTools(@ptrCast(&dummy)));
    try std.testing.expect(vt.supports_vision != null);
    try std.testing.expect(!vt.supports_vision.?(@ptrCast(&dummy)));
}

test "GeminiCliProvider supportsNativeTools returns false" {
    const vt = GeminiCliProvider.vtable;
    var dummy: u8 = 0;
    try std.testing.expect(!vt.supportsNativeTools(@ptrCast(&dummy)));
}

test "extractLastUserMessage finds last user" {
    const msgs = [_]ChatMessage{
        ChatMessage.system("Be helpful"),
        ChatMessage.user("first"),
        ChatMessage.assistant("ok"),
        ChatMessage.user("second"),
    };
    const result = extractLastUserMessage(&msgs);
    try std.testing.expectEqualStrings("second", result.?);
}

test "extractLastUserMessage returns null for no user" {
    const msgs = [_]ChatMessage{
        ChatMessage.system("Be helpful"),
        ChatMessage.assistant("ok"),
    };
    try std.testing.expect(extractLastUserMessage(&msgs) == null);
}

test "extractLastUserMessage empty messages" {
    const msgs = [_]ChatMessage{};
    try std.testing.expect(extractLastUserMessage(&msgs) == null);
}

test "checkCliVersion returns CliNotFound for missing binary" {
    const result = checkCliVersion(std.testing.allocator, "nonexistent_binary_xyzzy_gemini_99999");
    try std.testing.expectError(error.CliNotFound, result);
}
