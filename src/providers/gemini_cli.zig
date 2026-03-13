const std = @import("std");
const builtin = @import("builtin");
const root = @import("root.zig");

const Provider = root.Provider;
const ChatRequest = root.ChatRequest;
const ChatResponse = root.ChatResponse;
const ChatMessage = root.ChatMessage;

const log = std.log.scoped(.gemini_cli);

/// Provider that delegates to the `gemini` CLI (Google Gemini).
///
/// Runs `gemini --experimental-acp --approval-mode yolo -p <prompt> --model <model>
///            [--resume <session_id>]`
/// and parses the Agent Client Protocol (ACP) stream-json output for the
/// response text.  ACP events are newline-delimited JSON objects.
/// The provider first looks for a `{"type":"result","result":"..."}` event
/// and falls back to accumulating every `content` / `text` field in
/// `{"type":"content","content":"..."}` or `{"type":"text","text":"..."}` events.
///
/// `--approval-mode yolo` is required for headless (non-interactive) use so
/// that tool-use approvals are never blocked waiting for user input.
///
/// Set `session_id` on the struct before calling `chat*` to resume a previous
/// Gemini session.  Pass `"latest"` to resume the most recent session.
/// Use `listSessions` to enumerate available session IDs.
pub const GeminiCliProvider = struct {
    allocator: std.mem.Allocator,
    model: []const u8,
    /// Optional session ID for `--resume`.  Set to `"latest"` to resume the
    /// most recent session, or to a specific session ID from `listSessions`.
    session_id: ?[]const u8 = null,

    const DEFAULT_MODEL = "gemini-2.5-pro";
    const CLI_NAME = "gemini";
    const TIMEOUT_NS: u64 = 120 * std.time.ns_per_s;

    pub fn init(allocator: std.mem.Allocator, model: ?[]const u8) !GeminiCliProvider {
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
        .supports_vision = supportsVisionImpl,
        .getName = getNameImpl,
        .deinit = deinitImpl,
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

        // Combine system prompt with message if provided
        const prompt = if (system_prompt) |sys|
            try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ sys, message })
        else
            try allocator.dupe(u8, message);
        defer allocator.free(prompt);

        return runGemini(allocator, prompt, effective_model, self.session_id);
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

        // Extract last user message as prompt
        const prompt = extractLastUserMessage(request.messages) orelse return error.NoUserMessage;
        const content = try runGemini(allocator, prompt, effective_model, self.session_id);
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

    fn deinitImpl(_: *anyopaque) void {}

    /// Run the gemini CLI and parse ACP JSON output.
    ///
    /// Always passes `--approval-mode yolo` so the CLI never blocks on tool
    /// approval prompts when running headlessly.  Passes `--resume <session_id>`
    /// when `session_id` is non-null to continue an existing conversation.
    fn runGemini(
        allocator: std.mem.Allocator,
        prompt: []const u8,
        model: []const u8,
        session_id: ?[]const u8,
    ) ![]const u8 {
        // Build argv with optional --resume flag.
        var argv_list: std.ArrayListUnmanaged([]const u8) = .empty;
        defer argv_list.deinit(allocator);
        // 8 base args + up to 2 for --resume <session_id>
        try argv_list.ensureTotalCapacity(allocator, 10);
        try argv_list.append(allocator, CLI_NAME);
        try argv_list.append(allocator, "--experimental-acp");
        try argv_list.append(allocator, "--approval-mode");
        try argv_list.append(allocator, "yolo");
        try argv_list.append(allocator, "-p");
        try argv_list.append(allocator, prompt);
        try argv_list.append(allocator, "--model");
        try argv_list.append(allocator, model);
        if (session_id) |sid| {
            try argv_list.append(allocator, "--resume");
            try argv_list.append(allocator, sid);
        }
        const argv = argv_list.items;

        var child = std.process.Child.init(argv, allocator);
        child.stdin_behavior = .Close;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Ignore;

        try child.spawn();

        // Read all stdout
        const max_output: usize = 4 * 1024 * 1024; // 4 MB
        const stdout_result = child.stdout.?.readToEndAlloc(allocator, max_output) catch |err| {
            _ = child.wait() catch {};
            log.err("gemini CLI stdout read failed: {s}", .{@errorName(err)});
            return err;
        };
        defer allocator.free(stdout_result);

        const term = try child.wait();
        switch (term) {
            .Exited => |code| {
                if (code != 0) {
                    log.err("gemini CLI exited with code {d}; stdout preview: {s}", .{
                        code,
                        stdout_result[0..@min(stdout_result.len, 256)],
                    });
                    return error.CliProcessFailed;
                }
            },
            else => |t| {
                log.err("gemini CLI terminated abnormally: {any}", .{t});
                return error.CliProcessFailed;
            },
        }

        // Parse ACP stream-json output
        return parseAcpOutput(allocator, stdout_result);
    }

    /// Return a list of available Gemini CLI sessions.
    ///
    /// Runs `gemini --list-sessions` and returns the raw output as a
    /// heap-allocated string.  Caller must free the result.
    /// Returns a placeholder string in test builds to avoid spawning processes.
    pub fn listSessions(allocator: std.mem.Allocator) ![]const u8 {
        if (builtin.is_test) return try allocator.dupe(u8, "test-session-1\n");

        const argv = [_][]const u8{ CLI_NAME, "--list-sessions" };
        var child = std.process.Child.init(&argv, allocator);
        child.stdin_behavior = .Close;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Ignore;

        try child.spawn();

        const max_output: usize = 64 * 1024;
        const out = child.stdout.?.readToEndAlloc(allocator, max_output) catch |err| {
            _ = child.wait() catch {};
            return err;
        };
        errdefer allocator.free(out);

        const term = try child.wait();
        switch (term) {
            .Exited => |code| if (code != 0) {
                log.warn("gemini --list-sessions exited with code {d}", .{code});
                return error.CliProcessFailed;
            },
            else => return error.CliProcessFailed,
        }
        return out;
    }

    /// Fetch available model names by running `gemini -p "/model"`.
    ///
    /// Parses model identifiers (tokens beginning with `"gemini-"`) from the
    /// ACP result.  Returns a heap-allocated slice of heap-allocated strings;
    /// caller must free each element and the slice itself.
    /// Returns an empty slice when the CLI is unavailable or returns no
    /// recognisable model names — the caller should fall back to a static list.
    /// In test builds the subprocess is skipped and an empty slice is returned.
    pub fn fetchModels(allocator: std.mem.Allocator) [][]const u8 {
        if (builtin.is_test) return &.{};
        return fetchModelsInternal(allocator) catch &.{};
    }

    fn fetchModelsInternal(allocator: std.mem.Allocator) ![][]const u8 {
        const argv = [_][]const u8{
            CLI_NAME, "--experimental-acp", "--approval-mode",
            "yolo",   "-p",                 "/model",
        };
        var child = std.process.Child.init(&argv, allocator);
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

        // Parse ACP result text.
        // On failure `parseAcpOutput` propagates the error before `result_text`
        // is ever assigned, so the `defer` below is never registered and there is
        // no allocation to free — the early return is safe.
        const result_text = parseAcpOutput(allocator, out) catch return &.{};
        defer allocator.free(result_text);

        // Extract tokens that look like Gemini model IDs ("gemini-*").
        var models: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer {
            for (models.items) |m| allocator.free(m);
            models.deinit(allocator);
        }

        var line_iter = std.mem.splitScalar(u8, result_text, '\n');
        while (line_iter.next()) |line| {
            var token_iter = std.mem.tokenizeScalar(u8, line, ' ');
            while (token_iter.next()) |tok| {
                const clean = std.mem.trim(u8, tok, " \t\r,;:");
                if (clean.len > "gemini-".len and std.mem.startsWith(u8, clean, "gemini-")) {
                    try models.append(allocator, try allocator.dupe(u8, clean));
                }
            }
        }

        return models.toOwnedSlice(allocator);
    }

    /// Parse Gemini ACP (Agent Client Protocol) newline-delimited JSON output.
    ///
    /// Priority order (first match wins for instant-result events; streaming
    /// events are accumulated in declaration order):
    ///   1. `{"type":"result","result":"..."}` — full response, returned immediately.
    ///   2. `{"type":"model_turn_complete","model_output":{"text":"..."}}` — returned immediately.
    ///   3. `{"type":"output","content":"..."}` — accumulated.
    ///   4. `{"type":"content","content":"..."}` — accumulated.
    ///   5. `{"type":"text","text":"..."}` — accumulated.
    ///   6. `{"type":"content_block_delta","delta":{"text":"..."}}` — accumulated.
    ///   7. If none of the above matched AND the output contained no valid JSON
    ///      at all, the raw trimmed output is returned as plain text (handles
    ///      CLI versions that emit plain text rather than ACP JSON).
    pub fn parseAcpOutput(allocator: std.mem.Allocator, output: []const u8) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(allocator);

        var any_json: bool = false;

        var lines = std.mem.splitScalar(u8, output, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) continue;

            const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch continue;
            defer parsed.deinit();

            if (parsed.value != .object) continue;
            any_json = true;
            const obj = parsed.value.object;

            const type_val = obj.get("type") orelse continue;
            if (type_val != .string) continue;
            const type_str = type_val.string;

            // First priority: direct result event
            if (std.mem.eql(u8, type_str, "result")) {
                if (obj.get("result")) |result_val| {
                    if (result_val == .string) {
                        return try allocator.dupe(u8, result_val.string);
                    }
                }
            }

            // model_turn_complete: {"type":"model_turn_complete","model_output":{"text":"..."}}
            if (std.mem.eql(u8, type_str, "model_turn_complete")) {
                if (obj.get("model_output")) |mo| {
                    if (mo == .object) {
                        if (mo.object.get("text")) |text_val| {
                            if (text_val == .string) {
                                return try allocator.dupe(u8, text_val.string);
                            }
                        }
                    }
                }
            }

            // Accumulate output and content events:
            // {"type":"output","content":"..."} or {"type":"content","content":"..."}
            if (std.mem.eql(u8, type_str, "output") or std.mem.eql(u8, type_str, "content")) {
                if (obj.get("content")) |content_val| {
                    if (content_val == .string) {
                        try buf.appendSlice(allocator, content_val.string);
                    }
                }
            }

            // Accumulate text events: {"type":"text","text":"..."}
            if (std.mem.eql(u8, type_str, "text")) {
                if (obj.get("text")) |text_val| {
                    if (text_val == .string) {
                        try buf.appendSlice(allocator, text_val.string);
                    }
                }
            }

            // Accumulate content_block_delta streaming chunks:
            // {"type":"content_block_delta","delta":{"text":"..."}}
            if (std.mem.eql(u8, type_str, "content_block_delta")) {
                if (obj.get("delta")) |delta| {
                    if (delta == .object) {
                        if (delta.object.get("text")) |text_val| {
                            if (text_val == .string) {
                                try buf.appendSlice(allocator, text_val.string);
                            }
                        }
                    }
                }
            }
        }

        if (buf.items.len > 0) {
            return try buf.toOwnedSlice(allocator);
        }

        // Fallback: if the output contained no JSON at all, treat the raw
        // trimmed output as plain text.  This handles CLI versions that do not
        // emit ACP-formatted JSON.
        if (!any_json) {
            const trimmed = std.mem.trim(u8, output, " \t\n\r");
            if (trimmed.len > 0) {
                return try allocator.dupe(u8, trimmed);
            }
        }

        // Log a preview of the raw output to aid diagnosis before surfacing the error.
        // Guard with `builtin.is_test` so the diagnostic never fires during unit
        // tests (where deliberate bad-input tests would otherwise produce
        // "logged errors" failures in the Zig test harness).
        if (!builtin.is_test) log.warn("NoResultInOutput: no recognisable ACP event in output (any_json={}, len={d}); preview: {s}", .{
            any_json,
            output.len,
            output[0..@min(output.len, 512)],
        });
        return error.NoResultInOutput;
    }

    /// Health check: run `gemini --version` and verify exit code 0.
    fn healthCheck(allocator: std.mem.Allocator) !void {
        try checkCliVersion(allocator, CLI_NAME);
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Shared helpers
// ════════════════════════════════════════════════════════════════════════════

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
    switch (term) {
        .Exited => |code| {
            if (code != 0) return error.CliNotFound;
        },
        else => return error.CliNotFound,
    }
}

/// Extract the content of the last user message from a message slice.
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

test "parseAcpOutput extracts direct result event" {
    const input =
        \\{"type":"init","sessionId":"abc123","model":"gemini-2.5-pro"}
        \\{"type":"content","content":"partial text"}
        \\{"type":"result","result":"Hello from Gemini CLI!"}
    ;
    const result = try GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("Hello from Gemini CLI!", result);
}

test "parseAcpOutput accumulates content events when no result event" {
    const input =
        \\{"type":"init","sessionId":"abc123"}
        \\{"type":"content","content":"Hello "}
        \\{"type":"content","content":"world!"}
    ;
    const result = try GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("Hello world!", result);
}

test "parseAcpOutput accumulates text events when no result event" {
    const input =
        \\{"type":"init","sessionId":"abc123"}
        \\{"type":"text","text":"Hello "}
        \\{"type":"text","text":"world!"}
    ;
    const result = try GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("Hello world!", result);
}

test "parseAcpOutput result event takes priority over accumulated content" {
    const input =
        \\{"type":"content","content":"partial"}
        \\{"type":"result","result":"full response"}
    ;
    const result = try GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("full response", result);
}

test "parseAcpOutput no result returns error" {
    const input =
        \\{"type":"init","sessionId":"abc123"}
        \\{"type":"done","exitCode":0}
    ;
    const result = GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    try std.testing.expectError(error.NoResultInOutput, result);
}

test "parseAcpOutput handles empty input" {
    const result = GeminiCliProvider.parseAcpOutput(std.testing.allocator, "");
    try std.testing.expectError(error.NoResultInOutput, result);
}

test "parseAcpOutput handles invalid json lines gracefully" {
    const input =
        \\not json at all
        \\{"type":"result","result":"found it"}
    ;
    const result = try GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("found it", result);
}

test "parseAcpOutput skips result with non-string value" {
    const input =
        \\{"type":"result","result":42}
    ;
    const result = GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    try std.testing.expectError(error.NoResultInOutput, result);
}

test "parseAcpOutput model_turn_complete event" {
    const input =
        \\{"type":"init","sessionId":"s1"}
        \\{"type":"model_turn_complete","model_output":{"text":"Final answer"}}
    ;
    const result = try GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("Final answer", result);
}

test "parseAcpOutput output event accumulates" {
    const input =
        \\{"type":"output","content":"Hello "}
        \\{"type":"output","content":"world!"}
    ;
    const result = try GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("Hello world!", result);
}

test "parseAcpOutput content_block_delta streaming chunks" {
    const input =
        \\{"type":"content_block_delta","delta":{"text":"chunk1 "}}
        \\{"type":"content_block_delta","delta":{"text":"chunk2"}}
    ;
    const result = try GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("chunk1 chunk2", result);
}

test "parseAcpOutput plain text fallback when no json found" {
    const result = try GeminiCliProvider.parseAcpOutput(std.testing.allocator, "Just plain text response");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("Just plain text response", result);
}

test "parseAcpOutput plain text fallback trims whitespace" {
    const result = try GeminiCliProvider.parseAcpOutput(std.testing.allocator, "  trimmed  \n");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("trimmed", result);
}

test "parseAcpOutput no result returns error when json present but unrecognized" {
    // JSON is present (so any_json=true) but no result-bearing event found →
    // should NOT fall back to plain text, must return error.
    const input =
        \\{"type":"init","sessionId":"abc123"}
        \\{"type":"done","exitCode":0}
    ;
    const res = GeminiCliProvider.parseAcpOutput(std.testing.allocator, input);
    try std.testing.expectError(error.NoResultInOutput, res);
}

test "GeminiCliProvider default model is gemini-2.5-pro" {
    try std.testing.expectEqualStrings("gemini-2.5-pro", GeminiCliProvider.DEFAULT_MODEL);
}

test "GeminiCliProvider.listSessions returns placeholder in test builds" {
    const result = try GeminiCliProvider.listSessions(std.testing.allocator);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("test-session-1\n", result);
}

test "GeminiCliProvider.fetchModels returns empty slice in test builds" {
    const models = GeminiCliProvider.fetchModels(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), models.len);
}

test "GeminiCliProvider session_id field defaults to null" {
    // Just check the field exists and defaults to null without spawning real CLI.
    var dummy = GeminiCliProvider{
        .allocator = std.testing.allocator,
        .model = "gemini-2.5-pro",
    };
    try std.testing.expect(dummy.session_id == null);
    dummy.session_id = "latest";
    try std.testing.expectEqualStrings("latest", dummy.session_id.?);
}

test "checkCliVersion returns CliNotFound for missing binary" {
    const result = checkCliVersion(std.testing.allocator, "nonexistent_binary_xyzzy_gemini_99999");
    try std.testing.expectError(error.CliNotFound, result);
}

test "subprocess stderr_behavior Ignore does not deadlock on stderr output" {
    // Regression test for the bug where stderr_behavior = .Pipe without
    // reading stderr would deadlock if the child filled the OS pipe buffer.
    // With stderr_behavior = .Ignore the child writes freely without blocking.
    const argv = [_][]const u8{ "sh", "-c", "printf 'stderr output' >&2" };
    var child = std.process.Child.init(&argv, std.testing.allocator);
    child.stdin_behavior = .Close;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Ignore;
    try child.spawn();
    const out = try child.stdout.?.readToEndAlloc(std.testing.allocator, 4096);
    defer std.testing.allocator.free(out);
    const term = try child.wait();
    switch (term) {
        .Exited => |code| try std.testing.expectEqual(@as(u8, 0), code),
        else => return error.UnexpectedTermination,
    }
}

test "subprocess stdin_behavior Close gives child immediate EOF on stdin" {
    // Regression test for the bug where stdin_behavior = .Inherit let the
    // gemini CLI block reading interactive input (auth prompts, confirmations).
    // With stdin_behavior = .Close the child receives EOF immediately.
    const argv = [_][]const u8{ "sh", "-c", "read -r line; printf 'read:%s' \"$line\"" };
    var child = std.process.Child.init(&argv, std.testing.allocator);
    child.stdin_behavior = .Close;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Ignore;
    try child.spawn();
    const out = try child.stdout.?.readToEndAlloc(std.testing.allocator, 4096);
    defer std.testing.allocator.free(out);
    _ = try child.wait();
    // EOF on stdin causes 'read' to set line="" and exit; 'printf' outputs "read:"
    try std.testing.expectEqualStrings("read:", out);
}
