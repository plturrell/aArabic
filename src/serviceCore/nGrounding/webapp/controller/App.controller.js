sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment"
], function (Controller, JSONModel, MessageToast, MessageBox, Fragment) {
    "use strict";

    return Controller.extend("nLeanProof.webapp.controller.App", {

        _oAbortController: null,

        onInit: function () {
            // Main Model for Code and Output
            var oMainModel = new JSONModel({
                code: "-- Write your Lean4 code here\n-- Example: Natural number proof\n\ndef double (n : Nat) : Nat := n + n\n\n#check double 5\n#eval double 21",
                output: "",
                outputHtml: "<pre class='outputPre'>Ready. Click Check, Run, or Elaborate to process your code.</pre>",
                outputStatus: "info",
                lastOperation: "",
                busy: false,
                fileName: "untitled.lean",
                editorTheme: "default",
                lineCount: 7,
                errorCount: 0
            });
            this.getView().setModel(oMainModel, "main");

            // Chat Model
            var oChatModel = new JSONModel({
                messages: [],
                newMessage: "",
                busy: false,
                isStreaming: false,
                streamEnabled: true,
                messageCount: 0
            });
            this.getView().setModel(oChatModel, "chat");

            // Config Model (Server URLs)
            var oConfigModel = new JSONModel({
                leanServerUrl: "http://localhost:8001",
                aiServerUrl: "http://localhost:8001",
                serverConnected: false,
                apiKey: "",
                model: "lfm-2.5-1b",
                maxTokens: 2048
            });
            this.getView().setModel(oConfigModel, "config");

            // Check server connectivity
            this._checkServerHealth();

            // Load saved settings from localStorage
            this._loadSettings();

            // Initialize keyboard shortcuts
            this._initKeyboardShortcuts();
        },

        // --- Keyboard Shortcuts ---
        _initKeyboardShortcuts: function () {
            var that = this;
            this._fnKeyDownHandler = function (oEvent) {
                that._onKeyDown(oEvent);
            };
            document.addEventListener("keydown", this._fnKeyDownHandler);
        },

        _onKeyDown: function (oEvent) {
            var bCtrlOrCmd = oEvent.ctrlKey || oEvent.metaKey;
            var bShift = oEvent.shiftKey;
            var sKey = oEvent.key.toLowerCase();

            // Escape: Stop streaming (if streaming is active)
            if (sKey === "escape") {
                var oChatModel = this.getView().getModel("chat");
                if (oChatModel && oChatModel.getProperty("/isStreaming")) {
                    oEvent.preventDefault();
                    this.onStopStreaming();
                    return;
                }
            }

            if (!bCtrlOrCmd) {
                return;
            }

            // Ctrl+Enter or Cmd+Enter: Run code
            if (sKey === "enter" && !bShift) {
                oEvent.preventDefault();
                this.onRunCode();
                return;
            }

            // Ctrl+Shift+C or Cmd+Shift+C: Check code
            if (sKey === "c" && bShift) {
                oEvent.preventDefault();
                this.onCheckCode();
                return;
            }

            // Ctrl+Shift+E or Cmd+Shift+E: Elaborate code
            if (sKey === "e" && bShift) {
                oEvent.preventDefault();
                this.onElaborateCode();
                return;
            }

            // Ctrl+S or Cmd+S: Save file
            if (sKey === "s" && !bShift) {
                oEvent.preventDefault();
                this.onSaveFile();
                return;
            }

            // Ctrl+O or Cmd+O: Open file
            if (sKey === "o" && !bShift) {
                oEvent.preventDefault();
                this.onOpenFile();
                return;
            }

            // Ctrl+N or Cmd+N: New file
            if (sKey === "n" && !bShift) {
                oEvent.preventDefault();
                this.onNewFile();
                return;
            }
        },

        onExit: function () {
            // Clean up keyboard event listener
            if (this._fnKeyDownHandler) {
                document.removeEventListener("keydown", this._fnKeyDownHandler);
            }
        },

        // --- Server Health Check ---
        _checkServerHealth: function () {
            var oConfigModel = this.getView().getModel("config");
            var sUrl = oConfigModel.getProperty("/leanServerUrl") + "/health";

            fetch(sUrl, { method: "GET" })
                .then(response => {
                    oConfigModel.setProperty("/serverConnected", response.ok);
                })
                .catch(() => {
                    oConfigModel.setProperty("/serverConnected", false);
                });
        },

        // --- Settings Persistence ---
        _loadSettings: function () {
            try {
                var savedSettings = localStorage.getItem("nLeanProof_settings");
                if (savedSettings) {
                    var settings = JSON.parse(savedSettings);
                    var oConfigModel = this.getView().getModel("config");
                    if (settings.leanServerUrl) oConfigModel.setProperty("/leanServerUrl", settings.leanServerUrl);
                    if (settings.aiServerUrl) oConfigModel.setProperty("/aiServerUrl", settings.aiServerUrl);
                    if (settings.model) oConfigModel.setProperty("/model", settings.model);
                    if (settings.maxTokens) oConfigModel.setProperty("/maxTokens", settings.maxTokens);
                }
            } catch (e) {
                console.warn("Could not load settings:", e);
            }
        },

        _saveSettings: function () {
            try {
                var oConfigModel = this.getView().getModel("config");
                var settings = {
                    leanServerUrl: oConfigModel.getProperty("/leanServerUrl"),
                    aiServerUrl: oConfigModel.getProperty("/aiServerUrl"),
                    model: oConfigModel.getProperty("/model"),
                    maxTokens: oConfigModel.getProperty("/maxTokens")
                };
                localStorage.setItem("nLeanProof_settings", JSON.stringify(settings));
            } catch (e) {
                console.warn("Could not save settings:", e);
            }
        },

        // --- File Operations ---
        onNewFile: function () {
            var oMainModel = this.getView().getModel("main");
            MessageBox.confirm("Create a new file? Unsaved changes will be lost.", {
                onClose: function (oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        oMainModel.setProperty("/code", "-- New Lean4 file\n");
                        oMainModel.setProperty("/fileName", "untitled.lean");
                        oMainModel.setProperty("/output", "");
                        oMainModel.setProperty("/outputHtml", "<pre class='outputPre'>New file created.</pre>");
                    }
                }
            });
        },

        onOpenFile: function () {
            var that = this;
            var oFileInput = document.createElement("input");
            oFileInput.type = "file";
            oFileInput.accept = ".lean,.lean4";
            oFileInput.onchange = function (e) {
                var file = e.target.files[0];
                if (file) {
                    var reader = new FileReader();
                    reader.onload = function (evt) {
                        var oMainModel = that.getView().getModel("main");
                        oMainModel.setProperty("/code", evt.target.result);
                        oMainModel.setProperty("/fileName", file.name);
                        MessageToast.show("File loaded: " + file.name);
                    };
                    reader.readAsText(file);
                }
            };
            oFileInput.click();
        },

        onSaveFile: function () {
            var oMainModel = this.getView().getModel("main");
            var sCode = oMainModel.getProperty("/code");
            var sFileName = oMainModel.getProperty("/fileName");

            var blob = new Blob([sCode], { type: "text/plain" });
            var link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = sFileName;
            link.click();
            URL.revokeObjectURL(link.href);

            MessageToast.show("File saved: " + sFileName);
        },

        // --- Lean4 Operations ---
        onCheckCode: function () {
            this._callLeanEndpoint("/v1/lean4/check", "Check", "check");
        },

        onRunCode: function () {
            this._callLeanEndpoint("/v1/lean4/run", "Run", "run");
        },

        onElaborateCode: function () {
            this._callLeanEndpoint("/v1/lean4/elaborate", "Elaborate", "elaborate");
        },

        _callLeanEndpoint: function (sEndpoint, sOperation, sType) {
            var oMainModel = this.getView().getModel("main");
            var oConfigModel = this.getView().getModel("config");
            var sCode = oMainModel.getProperty("/code");
            var sUrl = oConfigModel.getProperty("/leanServerUrl") + sEndpoint;
            var that = this;

            // Update line count before processing
            this._updateLineCount();

            oMainModel.setProperty("/busy", true);
            oMainModel.setProperty("/lastOperation", sOperation + "...");
            oMainModel.setProperty("/outputHtml", "<pre class='outputPre processing'>Processing...</pre>");

            fetch(sUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source: sCode })
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => { throw new Error(text || response.statusText); });
                }
                return response.json();
            })
            .then(data => {
                var sOutput = that._formatLeanOutput(data, sType);
                oMainModel.setProperty("/output", JSON.stringify(data, null, 2));
                oMainModel.setProperty("/outputHtml", sOutput);
                oMainModel.setProperty("/outputStatus", data.success ? "success" : "error");
                oMainModel.setProperty("/lastOperation", sOperation + " complete");

                // Update error count from diagnostics
                var iErrorCount = 0;
                if (data.diagnostics && data.diagnostics.length > 0) {
                    iErrorCount = data.diagnostics.filter(function(d) { return d.severity === "error"; }).length;
                }
                if (!data.success && iErrorCount === 0) {
                    iErrorCount = 1;
                }
                oMainModel.setProperty("/errorCount", iErrorCount);

                MessageToast.show(sOperation + " successful");
            })
            .catch(err => {
                var sErrorHtml = "<pre class='outputPre error'>Error: " + that._escapeHtml(err.message) + "</pre>";
                oMainModel.setProperty("/output", "Error: " + err.message);
                oMainModel.setProperty("/outputHtml", sErrorHtml);
                oMainModel.setProperty("/outputStatus", "error");
                oMainModel.setProperty("/lastOperation", sOperation + " failed");
                oMainModel.setProperty("/errorCount", 1);
                MessageBox.error("Operation failed: " + err.message);
            })
            .finally(() => {
                oMainModel.setProperty("/busy", false);
            });
        },

        _updateLineCount: function () {
            var oMainModel = this.getView().getModel("main");
            var sCode = oMainModel.getProperty("/code") || "";
            var iLineCount = sCode.split("\n").length;
            oMainModel.setProperty("/lineCount", iLineCount);
        },

        _formatLeanOutput: function (data, sType) {
            var html = "<div class='leanOutput'>";

            if (data.success) {
                html += "<div class='outputSuccess'><span class='successIcon'>✓</span> " + sType.charAt(0).toUpperCase() + sType.slice(1) + " successful</div>";
            } else {
                html += "<div class='outputError'><span class='errorIcon'>✗</span> " + sType.charAt(0).toUpperCase() + sType.slice(1) + " failed</div>";
            }

            if (data.stdout) {
                html += "<pre class='outputPre stdout'>" + this._escapeHtml(data.stdout) + "</pre>";
            }
            if (data.stderr) {
                html += "<pre class='outputPre stderr'>" + this._escapeHtml(data.stderr) + "</pre>";
            }
            if (data.diagnostics && data.diagnostics.length > 0) {
                html += "<div class='diagnostics'>";
                data.diagnostics.forEach(function(d) {
                    var cls = d.severity === "error" ? "diagError" : (d.severity === "warning" ? "diagWarning" : "diagInfo");
                    html += "<div class='" + cls + "'>" + d.message + "</div>";
                });
                html += "</div>";
            }
            if (data.type) {
                html += "<div class='typeInfo'><strong>Type:</strong> " + this._escapeHtml(data.type) + "</div>";
            }
            if (!data.stdout && !data.stderr && !data.diagnostics && data.success) {
                html += "<pre class='outputPre'>No output.</pre>";
            }

            html += "</div>";
            return html;
        },

        _escapeHtml: function (text) {
            if (!text) return "";
            return text.replace(/&/g, "&amp;")
                       .replace(/</g, "&lt;")
                       .replace(/>/g, "&gt;")
                       .replace(/"/g, "&quot;")
                       .replace(/'/g, "&#039;");
        },

        // --- Output Operations ---
        onCopyOutput: function () {
            var oMainModel = this.getView().getModel("main");
            var sOutput = oMainModel.getProperty("/output");
            navigator.clipboard.writeText(sOutput).then(function () {
                MessageToast.show("Output copied to clipboard");
            });
        },

        onClearOutput: function () {
            var oMainModel = this.getView().getModel("main");
            oMainModel.setProperty("/output", "");
            oMainModel.setProperty("/outputHtml", "<pre class='outputPre'>Output cleared.</pre>");
            oMainModel.setProperty("/lastOperation", "");
        },

        // --- AI Operations ---
        onSendMessage: function () {
            var oChatModel = this.getView().getModel("chat");
            var sMessage = oChatModel.getProperty("/newMessage");
            if (!sMessage || !sMessage.trim()) return;

            var bStream = oChatModel.getProperty("/streamEnabled");

            // Add user message
            this._addChatMessage("user", sMessage);
            oChatModel.setProperty("/newMessage", "");
            oChatModel.setProperty("/busy", true);

            if (bStream) {
                this._sendStreamingMessage(sMessage);
            } else {
                this._sendNonStreamingMessage(sMessage);
            }
        },

        _addChatMessage: function (sRole, sContent) {
            var oChatModel = this.getView().getModel("chat");
            var aMessages = oChatModel.getProperty("/messages").slice();
            var sTimestamp = new Date().toLocaleTimeString();

            aMessages.push({
                role: sRole,
                content: sContent,
                contentHtml: this._formatChatContent(sContent),
                timestamp: sTimestamp
            });
            oChatModel.setProperty("/messages", aMessages);
            oChatModel.setProperty("/messageCount", aMessages.length);
        },

        _formatChatContent: function (sContent) {
            // Basic markdown-like formatting
            var html = this._escapeHtml(sContent);
            // Code blocks
            html = html.replace(/```(\w*)\n([\s\S]*?)```/g, "<pre class='chatCode'><code>$2</code></pre>");
            // Inline code
            html = html.replace(/`([^`]+)`/g, "<code class='inlineCode'>$1</code>");
            // Bold
            html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
            // Line breaks
            html = html.replace(/\n/g, "<br/>");
            return html;
        },

        _buildMessages: function () {
            var oChatModel = this.getView().getModel("chat");
            var oMainModel = this.getView().getModel("main");
            var aMessages = oChatModel.getProperty("/messages");

            var sCodeContext = oMainModel.getProperty("/code");
            var sSystemPrompt = "You are a helpful assistant for the Lean4 programming language and theorem proving. " +
                "The user is working on the following Lean4 code:\n\n```lean\n" + sCodeContext + "\n```\n\n" +
                "Help them understand concepts, fix errors, prove theorems, and write better Lean4 code.";

            var messages = [{ role: "system", content: sSystemPrompt }];
            aMessages.forEach(function (m) {
                messages.push({ role: m.role, content: m.content });
            });
            return messages;
        },

        _sendNonStreamingMessage: function (sUserMessage) {
            var that = this;
            var oChatModel = this.getView().getModel("chat");
            var oConfigModel = this.getView().getModel("config");
            var sUrl = oConfigModel.getProperty("/aiServerUrl") + "/v1/chat/completions";

            fetch(sUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: oConfigModel.getProperty("/model"),
                    messages: this._buildMessages(),
                    max_tokens: oConfigModel.getProperty("/maxTokens"),
                    stream: false
                })
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => { throw new Error(text || response.statusText); });
                }
                return response.json();
            })
            .then(data => {
                var sResponse = data.choices && data.choices[0] ? data.choices[0].message.content : "(No response)";
                that._addChatMessage("assistant", sResponse);
            })
            .catch(err => {
                that._addChatMessage("assistant", "Error: " + err.message);
            })
            .finally(() => {
                oChatModel.setProperty("/busy", false);
            });
        },

        _sendStreamingMessage: function (sUserMessage) {
            var that = this;
            var oChatModel = this.getView().getModel("chat");
            var oConfigModel = this.getView().getModel("config");
            var sUrl = oConfigModel.getProperty("/aiServerUrl") + "/v1/chat/completions";

            this._oAbortController = new AbortController();
            oChatModel.setProperty("/isStreaming", true);

            // Add placeholder for assistant response
            var aMessages = oChatModel.getProperty("/messages").slice();
            var nAssistantIdx = aMessages.length;
            aMessages.push({
                role: "assistant",
                content: "",
                contentHtml: "<span class='streamingCursor'>▌</span>",
                timestamp: new Date().toLocaleTimeString()
            });
            oChatModel.setProperty("/messages", aMessages);

            fetch(sUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: oConfigModel.getProperty("/model"),
                    messages: this._buildMessages(),
                    max_tokens: oConfigModel.getProperty("/maxTokens"),
                    stream: true
                }),
                signal: this._oAbortController.signal
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => { throw new Error(text || response.statusText); });
                }
                return that._processStream(response.body.getReader(), nAssistantIdx);
            })
            .catch(err => {
                if (err.name !== "AbortError") {
                    var aCurrentMessages = oChatModel.getProperty("/messages").slice();
                    aCurrentMessages[nAssistantIdx].content = "Error: " + err.message;
                    aCurrentMessages[nAssistantIdx].contentHtml = that._formatChatContent("Error: " + err.message);
                    oChatModel.setProperty("/messages", aCurrentMessages);
                }
            })
            .finally(() => {
                oChatModel.setProperty("/busy", false);
                oChatModel.setProperty("/isStreaming", false);
                that._oAbortController = null;
            });
        },

        _processStream: async function (reader, nAssistantIdx) {
            var oChatModel = this.getView().getModel("chat");
            var decoder = new TextDecoder();
            var sContent = "";
            var buffer = "";

            while (true) {
                var { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                var lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (var line of lines) {
                    if (line.startsWith("data: ")) {
                        var data = line.slice(6).trim();
                        if (data === "[DONE]") continue;
                        try {
                            var parsed = JSON.parse(data);
                            if (parsed.choices && parsed.choices[0].delta && parsed.choices[0].delta.content) {
                                sContent += parsed.choices[0].delta.content;
                                var aMessages = oChatModel.getProperty("/messages").slice();
                                aMessages[nAssistantIdx].content = sContent;
                                aMessages[nAssistantIdx].contentHtml = this._formatChatContent(sContent) + "<span class='streamingCursor'>▌</span>";
                                oChatModel.setProperty("/messages", aMessages);
                            }
                        } catch (e) { /* ignore parse errors */ }
                    }
                }
            }

            // Final update without cursor
            var aMessages = oChatModel.getProperty("/messages").slice();
            aMessages[nAssistantIdx].contentHtml = this._formatChatContent(sContent);
            oChatModel.setProperty("/messages", aMessages);
        },

        onStopStreaming: function () {
            if (this._oAbortController) {
                this._oAbortController.abort();
                MessageToast.show("Streaming stopped");
            }
        },

        onClearChat: function () {
            var oChatModel = this.getView().getModel("chat");
            oChatModel.setProperty("/messages", []);
            oChatModel.setProperty("/messageCount", 0);
            MessageToast.show("Chat cleared");
        },

        onFocusChat: function () {
            var oChatInput = this.byId("chatInput");
            if (oChatInput) {
                oChatInput.focus();
            }
        },

        onPressHome: function () {
            // Navigate back to launchpad
            var sLaunchpadUrl = "../nLaunchpad/webapp/index.html";
            window.location.href = sLaunchpadUrl;
        },

        onChatUpdateFinished: function () {
            // Scroll to bottom of chat
            var oChatList = this.byId("chatList");
            if (oChatList) {
                var oItems = oChatList.getItems();
                if (oItems.length > 0) {
                    oItems[oItems.length - 1].getDomRef()?.scrollIntoView({ behavior: "smooth" });
                }
            }
        },

        // --- Settings Dialog ---
        onSettingsPress: function () {
            var that = this;
            if (!this._oSettingsDialog) {
                this._oSettingsDialog = new sap.m.Dialog({
                    title: "Settings",
                    contentWidth: "400px",
                    content: [
                        new sap.m.VBox({
                            class: "sapUiSmallMargin",
                            items: [
                                new sap.m.Label({ text: "Lean4 Server URL", required: true }),
                                new sap.m.Input({ value: "{config>/leanServerUrl}", placeholder: "http://localhost:8001" }),
                                new sap.m.Label({ text: "AI Server URL", required: true, class: "sapUiSmallMarginTop" }),
                                new sap.m.Input({ value: "{config>/aiServerUrl}", placeholder: "http://localhost:8001" }),
                                new sap.m.Label({ text: "AI Model", class: "sapUiSmallMarginTop" }),
                                new sap.m.Input({ value: "{config>/model}", placeholder: "lfm-2.5-1b" }),
                                new sap.m.Label({ text: "Max Tokens", class: "sapUiSmallMarginTop" }),
                                new sap.m.StepInput({ value: "{config>/maxTokens}", min: 256, max: 8192, step: 256 }),
                                new sap.m.Label({ text: "Editor Theme", class: "sapUiSmallMarginTop" }),
                                new sap.m.Select({
                                    selectedKey: "{main>/editorTheme}",
                                    items: [
                                        new sap.ui.core.Item({ key: "default", text: "Default" }),
                                        new sap.ui.core.Item({ key: "hcb", text: "High Contrast Dark" }),
                                        new sap.ui.core.Item({ key: "hcw", text: "High Contrast Light" })
                                    ]
                                })
                            ]
                        })
                    ],
                    beginButton: new sap.m.Button({
                        text: "Save",
                        type: "Emphasized",
                        press: function () {
                            that._saveSettings();
                            that._checkServerHealth();
                            that._oSettingsDialog.close();
                            MessageToast.show("Settings saved");
                        }
                    }),
                    endButton: new sap.m.Button({
                        text: "Cancel",
                        press: function () {
                            that._oSettingsDialog.close();
                        }
                    })
                });
                this.getView().addDependent(this._oSettingsDialog);
            }
            this._oSettingsDialog.open();
        },

        onPressHome: function () {
            window.location.href = "http://localhost:8091";
        }
    });
});
