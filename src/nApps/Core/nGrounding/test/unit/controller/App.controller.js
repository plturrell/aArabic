/* global QUnit, sinon */

sap.ui.define([
    "nLeanProof/webapp/controller/App.controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageBox",
    "sap/m/MessageToast"
], function (AppController, JSONModel, MessageBox, MessageToast) {
    "use strict";

    QUnit.module("App Controller - onInit", {
        beforeEach: function () {
            this.oController = new AppController();
            
            // Mock view with models
            this.aModels = {};
            this.oController.getView = sinon.stub().returns({
                setModel: sinon.spy(function (oModel, sName) {
                    this.aModels[sName] = oModel;
                }.bind(this)),
                getModel: sinon.spy(function (sName) {
                    return this.aModels[sName];
                }.bind(this)),
                addDependent: sinon.stub()
            });

            // Mock localStorage
            this.oLocalStorageStub = sinon.stub(window.localStorage, "getItem").returns(null);
            this.oLocalStorageSetStub = sinon.stub(window.localStorage, "setItem");

            // Mock fetch for health check
            this.oFetchStub = sinon.stub(window, "fetch").resolves({
                ok: true,
                json: function () { return Promise.resolve({}); },
                text: function () { return Promise.resolve(""); }
            });
        },
        afterEach: function () {
            this.oLocalStorageStub.restore();
            this.oLocalStorageSetStub.restore();
            this.oFetchStub.restore();
            this.oController.destroy();
        }
    });

    QUnit.test("onInit creates main model with correct default values", function (assert) {
        // Act
        this.oController.onInit();

        // Assert
        var oMainModel = this.aModels["main"];
        assert.ok(oMainModel, "Main model should be created");
        assert.strictEqual(oMainModel.getProperty("/fileName"), "untitled.lean", "Default filename should be 'untitled.lean'");
        assert.strictEqual(oMainModel.getProperty("/busy"), false, "Busy should be false initially");
        assert.strictEqual(oMainModel.getProperty("/editorTheme"), "default", "Default theme should be 'default'");
    });

    QUnit.test("onInit creates chat model with correct structure", function (assert) {
        // Act
        this.oController.onInit();

        // Assert
        var oChatModel = this.aModels["chat"];
        assert.ok(oChatModel, "Chat model should be created");
        assert.deepEqual(oChatModel.getProperty("/messages"), [], "Messages should be empty array");
        assert.strictEqual(oChatModel.getProperty("/streamEnabled"), true, "Streaming should be enabled by default");
    });

    QUnit.test("onInit creates config model with server URLs", function (assert) {
        // Act
        this.oController.onInit();

        // Assert
        var oConfigModel = this.aModels["config"];
        assert.ok(oConfigModel, "Config model should be created");
        assert.strictEqual(oConfigModel.getProperty("/leanServerUrl"), "http://localhost:8001", "Lean server URL should be localhost:8001");
        assert.strictEqual(oConfigModel.getProperty("/model"), "lfm-2.5-1b", "Default model should be 'lfm-2.5-1b'");
    });

    // ============================================
    // File Operations Tests
    // ============================================
    QUnit.module("App Controller - File Operations", {
        beforeEach: function () {
            this.oController = new AppController();
            
            // Create models
            this.oMainModel = new JSONModel({
                code: "-- test code",
                fileName: "test.lean",
                output: "",
                outputHtml: ""
            });

            this.oController.getView = sinon.stub().returns({
                getModel: sinon.stub().callsFake(function (sName) {
                    if (sName === "main") return this.oMainModel;
                    return null;
                }.bind(this))
            });

            this.oMessageBoxStub = sinon.stub(MessageBox, "confirm");
            this.oMessageToastStub = sinon.stub(MessageToast, "show");
        },
        afterEach: function () {
            this.oMessageBoxStub.restore();
            this.oMessageToastStub.restore();
            this.oController.destroy();
        }
    });

    QUnit.test("onNewFile shows confirmation dialog", function (assert) {
        // Act
        this.oController.onNewFile();

        // Assert
        assert.ok(this.oMessageBoxStub.calledOnce, "MessageBox.confirm should be called");
        assert.ok(this.oMessageBoxStub.firstCall.args[0].includes("new file"), "Dialog message should mention new file");
    });

    QUnit.test("onNewFile resets model when confirmed", function (assert) {
        // Arrange
        this.oMessageBoxStub.callsFake(function (sMessage, oOptions) {
            oOptions.onClose(MessageBox.Action.OK);
        });

        // Act
        this.oController.onNewFile();

        // Assert
        assert.strictEqual(this.oMainModel.getProperty("/fileName"), "untitled.lean", "Filename should be reset");
        assert.ok(this.oMainModel.getProperty("/code").includes("New Lean4 file"), "Code should be reset");
    });

    QUnit.test("onSaveFile creates download link with correct filename", function (assert) {
        // Arrange
        var oCreateElementSpy = sinon.spy(document, "createElement");
        var oRevokeObjectURLStub = sinon.stub(URL, "revokeObjectURL");
        var oCreateObjectURLStub = sinon.stub(URL, "createObjectURL").returns("blob:test");

        // Act
        this.oController.onSaveFile();

        // Assert
        assert.ok(oCreateElementSpy.calledWith("a"), "Should create an anchor element");
        assert.ok(this.oMessageToastStub.calledOnce, "MessageToast should show success");

        // Cleanup
        oCreateElementSpy.restore();
        oRevokeObjectURLStub.restore();
        oCreateObjectURLStub.restore();
    });

    // ============================================
    // Lean4 Endpoint Tests
    // ============================================
    QUnit.module("App Controller - Lean4 Endpoint Calls", {
        beforeEach: function () {
            this.oController = new AppController();

            this.oMainModel = new JSONModel({
                code: "def test := 1",
                output: "",
                outputHtml: "",
                busy: false,
                lastOperation: ""
            });

            this.oConfigModel = new JSONModel({
                leanServerUrl: "http://localhost:8001"
            });

            this.oController.getView = sinon.stub().returns({
                getModel: sinon.stub().callsFake(function (sName) {
                    if (sName === "main") return this.oMainModel;
                    if (sName === "config") return this.oConfigModel;
                    return null;
                }.bind(this))
            });

            this.oFetchStub = sinon.stub(window, "fetch");
            this.oMessageToastStub = sinon.stub(MessageToast, "show");
            this.oMessageBoxStub = sinon.stub(MessageBox, "error");
        },
        afterEach: function () {
            this.oFetchStub.restore();
            this.oMessageToastStub.restore();
            this.oMessageBoxStub.restore();
            this.oController.destroy();
        }
    });

    QUnit.test("_callLeanEndpoint sets busy state during request", function (assert) {
        // Arrange
        var done = assert.async();
        this.oFetchStub.resolves({
            ok: true,
            json: function () {
                return Promise.resolve({ success: true, stdout: "ok" });
            }
        });

        // Act
        this.oController._callLeanEndpoint("/v1/lean4/check", "Check", "check");

        // Assert - busy should be true immediately
        assert.strictEqual(this.oMainModel.getProperty("/busy"), true, "Busy should be true during request");

        // Wait for async completion
        setTimeout(function () {
            assert.strictEqual(this.oMainModel.getProperty("/busy"), false, "Busy should be false after request");
            done();
        }.bind(this), 100);
    });

    QUnit.test("_callLeanEndpoint calls correct endpoint URL", function (assert) {
        // Arrange
        var done = assert.async();
        this.oFetchStub.resolves({
            ok: true,
            json: function () {
                return Promise.resolve({ success: true });
            }
        });

        // Act
        this.oController._callLeanEndpoint("/v1/lean4/check", "Check", "check");

        // Assert
        setTimeout(function () {
            assert.ok(this.oFetchStub.calledOnce, "Fetch should be called once");
            var sUrl = this.oFetchStub.firstCall.args[0];
            assert.strictEqual(sUrl, "http://localhost:8001/v1/lean4/check", "URL should be correct");
            done();
        }.bind(this), 50);
    });

    QUnit.test("_callLeanEndpoint handles error response", function (assert) {
        // Arrange
        var done = assert.async();
        this.oFetchStub.resolves({
            ok: false,
            statusText: "Internal Server Error",
            text: function () {
                return Promise.resolve("Server error details");
            }
        });

        // Act
        this.oController._callLeanEndpoint("/v1/lean4/check", "Check", "check");

        // Assert
        setTimeout(function () {
            assert.strictEqual(this.oMainModel.getProperty("/outputStatus"), "error", "Status should be error");
            assert.ok(this.oMessageBoxStub.called, "MessageBox.error should be called");
            done();
        }.bind(this), 100);
    });

    QUnit.test("_callLeanEndpoint handles network failure", function (assert) {
        // Arrange
        var done = assert.async();
        this.oFetchStub.rejects(new Error("Network error"));

        // Act
        this.oController._callLeanEndpoint("/v1/lean4/check", "Check", "check");

        // Assert
        setTimeout(function () {
            assert.strictEqual(this.oMainModel.getProperty("/outputStatus"), "error", "Status should be error");
            assert.ok(this.oMainModel.getProperty("/output").includes("Network error"), "Output should contain error message");
            done();
        }.bind(this), 100);
    });

    // ============================================
    // Chat Message Handling Tests
    // ============================================
    QUnit.module("App Controller - Chat Message Handling", {
        beforeEach: function () {
            this.oController = new AppController();

            this.oChatModel = new JSONModel({
                messages: [],
                newMessage: "",
                busy: false
            });

            this.oController.getView = sinon.stub().returns({
                getModel: sinon.stub().callsFake(function (sName) {
                    if (sName === "chat") return this.oChatModel;
                    return null;
                }.bind(this))
            });
        },
        afterEach: function () {
            this.oController.destroy();
        }
    });

    QUnit.test("_addChatMessage adds message to chat model", function (assert) {
        // Act
        this.oController._addChatMessage("user", "Hello world");

        // Assert
        var aMessages = this.oChatModel.getProperty("/messages");
        assert.strictEqual(aMessages.length, 1, "Should have one message");
        assert.strictEqual(aMessages[0].role, "user", "Role should be 'user'");
        assert.strictEqual(aMessages[0].content, "Hello world", "Content should match");
        assert.ok(aMessages[0].timestamp, "Should have timestamp");
    });

    QUnit.test("_addChatMessage adds multiple messages in order", function (assert) {
        // Act
        this.oController._addChatMessage("user", "Question");
        this.oController._addChatMessage("assistant", "Answer");

        // Assert
        var aMessages = this.oChatModel.getProperty("/messages");
        assert.strictEqual(aMessages.length, 2, "Should have two messages");
        assert.strictEqual(aMessages[0].role, "user", "First should be user");
        assert.strictEqual(aMessages[1].role, "assistant", "Second should be assistant");
    });

    QUnit.test("_formatChatContent escapes HTML and formats code", function (assert) {
        // Act
        var sResult = this.oController._formatChatContent("Use `code` here");

        // Assert
        assert.ok(sResult.includes("<code class='inlineCode'>code</code>"), "Should format inline code");
    });

    QUnit.test("_formatChatContent handles code blocks", function (assert) {
        // Act
        var sContent = "```lean\ndef test := 1\n```";
        var sResult = this.oController._formatChatContent(sContent);

        // Assert
        assert.ok(sResult.includes("<pre class='chatCode'>"), "Should have pre tag for code block");
        assert.ok(sResult.includes("<code>"), "Should have code tag");
    });

    QUnit.test("_formatChatContent converts bold text", function (assert) {
        // Act
        var sResult = this.oController._formatChatContent("This is **bold** text");

        // Assert
        assert.ok(sResult.includes("<strong>bold</strong>"), "Should convert bold markers to strong tags");
    });

    QUnit.test("_formatChatContent converts newlines to br tags", function (assert) {
        // Act
        var sResult = this.oController._formatChatContent("Line 1\nLine 2");

        // Assert
        assert.ok(sResult.includes("<br/>"), "Should convert newlines to br tags");
    });

    // ============================================
    // Settings Persistence Tests
    // ============================================
    QUnit.module("App Controller - Settings Persistence", {
        beforeEach: function () {
            this.oController = new AppController();

            this.oConfigModel = new JSONModel({
                leanServerUrl: "http://localhost:8001",
                aiServerUrl: "http://localhost:8001",
                model: "lfm-2.5-1b",
                maxTokens: 2048
            });

            this.oController.getView = sinon.stub().returns({
                getModel: sinon.stub().callsFake(function (sName) {
                    if (sName === "config") return this.oConfigModel;
                    return null;
                }.bind(this))
            });

            this.oGetItemStub = sinon.stub(window.localStorage, "getItem");
            this.oSetItemStub = sinon.stub(window.localStorage, "setItem");
        },
        afterEach: function () {
            this.oGetItemStub.restore();
            this.oSetItemStub.restore();
            this.oController.destroy();
        }
    });

    QUnit.test("_loadSettings loads saved settings from localStorage", function (assert) {
        // Arrange
        var oSavedSettings = {
            leanServerUrl: "http://custom:9000",
            aiServerUrl: "http://ai:9001",
            model: "custom-model",
            maxTokens: 4096
        };
        this.oGetItemStub.returns(JSON.stringify(oSavedSettings));

        // Act
        this.oController._loadSettings();

        // Assert
        assert.strictEqual(this.oConfigModel.getProperty("/leanServerUrl"), "http://custom:9000", "Lean URL should be loaded");
        assert.strictEqual(this.oConfigModel.getProperty("/model"), "custom-model", "Model should be loaded");
        assert.strictEqual(this.oConfigModel.getProperty("/maxTokens"), 4096, "Max tokens should be loaded");
    });

    QUnit.test("_loadSettings handles missing localStorage gracefully", function (assert) {
        // Arrange
        this.oGetItemStub.returns(null);

        // Act - should not throw
        this.oController._loadSettings();

        // Assert - model should retain default values
        assert.strictEqual(this.oConfigModel.getProperty("/leanServerUrl"), "http://localhost:8001", "Should keep default URL");
    });

    QUnit.test("_loadSettings handles invalid JSON gracefully", function (assert) {
        // Arrange
        this.oGetItemStub.returns("not valid json{");

        // Act - should not throw
        this.oController._loadSettings();

        // Assert - model should retain default values
        assert.strictEqual(this.oConfigModel.getProperty("/leanServerUrl"), "http://localhost:8001", "Should keep default URL on parse error");
    });

    QUnit.test("_saveSettings stores settings to localStorage", function (assert) {
        // Act
        this.oController._saveSettings();

        // Assert
        assert.ok(this.oSetItemStub.calledOnce, "setItem should be called once");
        assert.strictEqual(this.oSetItemStub.firstCall.args[0], "nLeanProof_settings", "Key should be correct");

        var sSavedValue = this.oSetItemStub.firstCall.args[1];
        var oSaved = JSON.parse(sSavedValue);
        assert.strictEqual(oSaved.leanServerUrl, "http://localhost:8001", "Lean URL should be saved");
        assert.strictEqual(oSaved.model, "lfm-2.5-1b", "Model should be saved");
    });

    QUnit.test("_saveSettings handles localStorage errors gracefully", function (assert) {
        // Arrange
        this.oSetItemStub.throws(new Error("QuotaExceeded"));

        // Act - should not throw
        this.oController._saveSettings();

        // Assert - no exception thrown
        assert.ok(true, "Should handle localStorage error without throwing");
    });

    // ============================================
    // HTML Escaping Tests
    // ============================================
    QUnit.module("App Controller - HTML Escaping", {
        beforeEach: function () {
            this.oController = new AppController();
        },
        afterEach: function () {
            this.oController.destroy();
        }
    });

    QUnit.test("_escapeHtml escapes ampersand", function (assert) {
        // Act
        var sResult = this.oController._escapeHtml("A & B");

        // Assert
        assert.strictEqual(sResult, "A &amp; B", "Ampersand should be escaped");
    });

    QUnit.test("_escapeHtml escapes less than and greater than", function (assert) {
        // Act
        var sResult = this.oController._escapeHtml("<script>alert('xss')</script>");

        // Assert
        assert.strictEqual(sResult, "&lt;script&gt;alert(&#039;xss&#039;)&lt;/script&gt;", "HTML tags should be escaped");
    });

    QUnit.test("_escapeHtml escapes quotes", function (assert) {
        // Act
        var sResult = this.oController._escapeHtml('He said "hello"');

        // Assert
        assert.strictEqual(sResult, "He said &quot;hello&quot;", "Double quotes should be escaped");
    });

    QUnit.test("_escapeHtml escapes single quotes", function (assert) {
        // Act
        var sResult = this.oController._escapeHtml("It's working");

        // Assert
        assert.strictEqual(sResult, "It&#039;s working", "Single quotes should be escaped");
    });

    QUnit.test("_escapeHtml handles null and empty strings", function (assert) {
        // Act & Assert
        assert.strictEqual(this.oController._escapeHtml(null), "", "Null should return empty string");
        assert.strictEqual(this.oController._escapeHtml(""), "", "Empty string should return empty string");
        assert.strictEqual(this.oController._escapeHtml(undefined), "", "Undefined should return empty string");
    });

    QUnit.test("_escapeHtml escapes all special characters together", function (assert) {
        // Act
        var sResult = this.oController._escapeHtml("<div class=\"test\" data-value='foo & bar'>");

        // Assert
        assert.strictEqual(
            sResult,
            "&lt;div class=&quot;test&quot; data-value=&#039;foo &amp; bar&#039;&gt;",
            "All special characters should be escaped together"
        );
    });

    QUnit.test("_escapeHtml leaves normal text unchanged", function (assert) {
        // Act
        var sResult = this.oController._escapeHtml("Normal text with numbers 123 and symbols @#$%");

        // Assert
        assert.strictEqual(sResult, "Normal text with numbers 123 and symbols @#$%", "Normal text should be unchanged");
    });
});

