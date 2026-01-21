/* global QUnit */

sap.ui.define([
    "test/unit/controller/App.controller"
], function () {
    "use strict";

    QUnit.config.autostart = false;

    sap.ui.require([
        "sap/ui/core/Core"
    ], function (Core) {
        Core.ready().then(function () {
            QUnit.start();
        });
    });
});

