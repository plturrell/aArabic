sap.ui.define([
    "sap/ui/core/ComponentContainer"
], function (ComponentContainer) {
    "use strict";

    // Initialize the UI component in the "content" div
    new ComponentContainer({
        name: "hypershimmy",
        settings: {
            id: "hypershimmy"
        },
        async: true
    }).placeAt("content");
});
