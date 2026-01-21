sap.ui.define([], function () {
    "use strict";
    return {
        fileIcon: function (sType) {
            const mIcons = {
                "Mojo Source": "sap-icon://document-text",
                "Zig Source": "sap-icon://source-code",
                "Markdown": "sap-icon://attachment-text",
                "JavaScript": "sap-icon://nodejs",
                "JSON": "sap-icon://syntax"
            };
            return mIcons[sType] || "sap-icon://document";
        },

        statusState: function (sStatus) {
            switch (sStatus) {
                case "modified": return "Warning";
                case "error": return "Error";
                case "new": return "Success";
                default: return "None";
            }
        }
    };
});
