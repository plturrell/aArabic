sap.ui.define([
    "sap/ui/core/ComponentContainer"
], function (ComponentContainer) {
    "use strict";

    new ComponentContainer({
        name: "trialbalance",
        settings: {
            id: "trialbalance"
        },
        async: true
    }).placeAt("content");

});