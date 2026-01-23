sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/ResponsivePopover",
    "sap/m/NotificationListGroup",
    "sap/m/Bar",
    "sap/m/Button",
    "sap/m/Title",
    "sap/m/Text",
    "sap/m/VBox",
    "llm/server/dashboard/utils/ThemeService",
    "llm/server/dashboard/utils/KeyboardShortcuts",
    "llm/server/dashboard/utils/NotificationService"
], function (Controller, ResponsivePopover, NotificationListGroup, Bar, Button, Title, Text, VBox,
             ThemeService, KeyboardShortcuts, NotificationService) {
    "use strict";

    return Controller.extend("llm.server.dashboard.controller.App", {

        _oNotificationPopover: null,

        onInit: function () {
            // Apply content density mode
            this.getView().addStyleClass(this.getOwnerComponent().getContentDensityClass());

            // Set initial selection in NavigationList
            var oSideNav = this.byId("sideNavigation");
            if (oSideNav) {
                var oNavList = oSideNav.getItem();
                if (oNavList && oNavList.getItems().length > 0) {
                    oNavList.setSelectedItem(oNavList.getItems()[0]);
                }
            }

            // Restore sidebar state from localStorage
            var bSidebarExpanded = this._getSidebarState();
            if (!bSidebarExpanded) {
                // If sidebar was collapsed, collapse it on init
                if (oSideNav) {
                    oSideNav.setExpanded(false);
                }
            }

            // Update hamburger icon based on initial state
            this._updateHamburgerIcon(bSidebarExpanded);

            // Initialize theme toggle button state
            this._initThemeToggle();

            // Initialize keyboard shortcuts
            KeyboardShortcuts.init(this);

            // Initialize notification service and badge
            this._initNotifications();
        },

        onExit: function () {
            // Cleanup keyboard shortcuts on controller exit
            KeyboardShortcuts.destroy();
        },

        _initThemeToggle: function () {
            var oThemeToggleBtn = this.byId("themeToggleBtn");
            if (oThemeToggleBtn) {
                var bIsDark = ThemeService.isDarkMode();
                oThemeToggleBtn.setPressed(bIsDark);
                this._updateThemeToggleIcon(bIsDark);
            }
        },

        onThemeToggle: function (oEvent) {
            var bPressed = oEvent.getParameter("pressed");
            var sNewTheme = bPressed ? ThemeService.THEMES.DARK : ThemeService.THEMES.LIGHT;
            ThemeService.setTheme(sNewTheme);
            this._updateThemeToggleIcon(bPressed);
        },

        _updateThemeToggleIcon: function (bIsDark) {
            var oThemeToggleBtn = this.byId("themeToggleBtn");
            if (oThemeToggleBtn) {
                var sIcon = bIsDark ? "sap-icon://dark-mode" : "sap-icon://light-mode";
                var sTooltip = bIsDark ? "Switch to Light Mode" : "Switch to Dark Mode";
                oThemeToggleBtn.setIcon(sIcon);
                oThemeToggleBtn.setTooltip(sTooltip);
            }
        },

        onToggleSidebar: function () {
            var oSideNav = this.byId("sideNavigation");
            var bCurrentlyExpanded = oSideNav.getExpanded();
            
            // Toggle the sidebar
            oSideNav.setExpanded(!bCurrentlyExpanded);
            
            // Update icon and save state
            this._updateHamburgerIcon(!bCurrentlyExpanded);
            this._saveSidebarState(!bCurrentlyExpanded);
        },
        
        _updateHamburgerIcon: function (bExpanded) {
            var oHamburgerBtn = this.byId("hamburgerBtn");
            if (oHamburgerBtn) {
                // Change icon based on state
                var sIcon = bExpanded ? "sap-icon://close-command-field" : "sap-icon://menu2";
                var sTooltip = bExpanded ? "Close Menu" : "Open Menu";
                oHamburgerBtn.setIcon(sIcon);
                oHamburgerBtn.setTooltip(sTooltip);
            }
        },
        
        _getSidebarState: function () {
            // Default to expanded (true) if no saved state
            var sSavedState = window.localStorage.getItem("sidebarExpanded");
            return sSavedState === null ? true : sSavedState === "true";
        },
        
        _saveSidebarState: function (bExpanded) {
            try {
                window.localStorage.setItem("sidebarExpanded", bExpanded.toString());
            } catch (e) {
                console.warn("Could not save sidebar state to localStorage:", e);
            }
        },
        
        onNavigationSelect: function (oEvent) {
            var oItem = oEvent.getParameter("item");
            var sKey = oItem.getKey();
            var oRouter = this.getOwnerComponent().getRouter();
            
            // Map keys to page IDs
            var mPageMap = {
                "main": "main",
                "promptTesting": "promptTesting",
                "mhcTuning": "mhcTuning",
                "orchestration": "orchestration",
                "modelVersions": "modelVersions",
                "trainingDashboard": "trainingDashboard",
                "modelRouter": "modelRouter",
                "abTesting": "abTesting",
                "settings": "settings"
            };
            
            var sRouteName = mPageMap[sKey];
            if (sRouteName && oRouter) {
                oRouter.navTo(sRouteName);
                console.log("Navigated to route:", sRouteName);
            }
            
            // On phone/tablet, auto-collapse sidebar after selection
            if (sap.ui.Device.system.phone || sap.ui.Device.system.tablet) {
                var oSideNav = this.byId("sideNavigation");
                if (oSideNav) {
                    oSideNav.setExpanded(false);
                    this._updateHamburgerIcon(false);
                    this._saveSidebarState(false);
                }
            }
        },

        onRefresh: function () {
            // Refresh current page
            window.location.reload();
        },

        onSettings: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            if (oRouter) {
                oRouter.navTo("settings");
            }

            // Update side navigation selection
            var oSideNav = this.byId("sideNavigation");
            if (oSideNav) {
                var oNavList = oSideNav.getItem();
                if (oNavList) {
                    var aItems = oNavList.getItems();
                    for (var i = 0; i < aItems.length; i++) {
                        if (aItems[i].getKey() === "settings") {
                            oNavList.setSelectedItem(aItems[i]);
                            break;
                        }
                    }
                }
            }
        },

        onNotifications: function (oEvent) {
            var oButton = oEvent.getSource();
            this._getNotificationPopover().openBy(oButton);
        },

        // ========================================================================
        // NOTIFICATION METHODS
        // ========================================================================

        _initNotifications: function () {
            var that = this;

            // Update badge on init
            this._updateNotificationBadge();

            // Listen for notification changes
            NotificationService.attachEvent("added", function () {
                that._updateNotificationBadge();
            });
            NotificationService.attachEvent("removed", function () {
                that._updateNotificationBadge();
            });
            NotificationService.attachEvent("changed", function () {
                that._updateNotificationBadge();
            });
            NotificationService.attachEvent("allChanged", function () {
                that._updateNotificationBadge();
            });
        },

        _updateNotificationBadge: function () {
            var oButton = this.byId("notificationBtn");
            if (oButton) {
                var iCount = NotificationService.getUnreadCount();
                var sTooltip = iCount > 0
                    ? "Notifications (" + iCount + " unread)"
                    : "Notifications";
                oButton.setTooltip(sTooltip);

                // Update custom data for badge
                var aCustomData = oButton.getCustomData();
                if (aCustomData && aCustomData.length > 0) {
                    aCustomData[0].setValue(iCount > 0 ? String(iCount) : "");
                }
            }
        },

        _getNotificationPopover: function () {
            if (!this._oNotificationPopover) {
                this._oNotificationPopover = this._createNotificationPopover();
            }
            this._refreshNotificationPopover();
            return this._oNotificationPopover;
        },

        _createNotificationPopover: function () {
            var that = this;

            var oPopover = new ResponsivePopover({
                title: "Notifications",
                contentWidth: "400px",
                contentHeight: "450px",
                placement: "Bottom",
                customHeader: new Bar({
                    contentLeft: [
                        new Title({ text: "Notifications" })
                    ],
                    contentRight: [
                        new Button({
                            text: "Mark All Read",
                            type: "Transparent",
                            press: function () {
                                NotificationService.markAllAsRead();
                                that._refreshNotificationPopover();
                            }
                        }),
                        new Button({
                            icon: "sap-icon://delete",
                            type: "Transparent",
                            tooltip: "Clear All",
                            press: function () {
                                NotificationService.clearAll();
                                that._refreshNotificationPopover();
                            }
                        })
                    ]
                }),
                content: [],
                afterClose: function () {
                    // Mark visible notifications as read when popover closes
                }
            });

            oPopover.addStyleClass(this.getOwnerComponent().getContentDensityClass());
            return oPopover;
        },

        _refreshNotificationPopover: function () {
            var that = this;
            var oPopover = this._oNotificationPopover;

            // Clear existing content
            oPopover.removeAllContent();

            var aNotifications = NotificationService.getNotifications();

            if (aNotifications.length === 0) {
                // Show empty state
                oPopover.addContent(new VBox({
                    alignItems: "Center",
                    justifyContent: "Center",
                    height: "100%",
                    items: [
                        new Text({ text: "No notifications" }).addStyleClass("sapUiMediumMarginTop")
                    ]
                }));
            } else {
                // Group by today/earlier
                var oToday = new Date();
                oToday.setHours(0, 0, 0, 0);

                var aTodayNotifs = [];
                var aEarlierNotifs = [];

                aNotifications.forEach(function (n) {
                    var oDate = new Date(n.timestamp);
                    if (oDate >= oToday) {
                        aTodayNotifs.push(n);
                    } else {
                        aEarlierNotifs.push(n);
                    }
                });

                // Create notification groups
                if (aTodayNotifs.length > 0) {
                    var oTodayGroup = new NotificationListGroup({
                        title: "Today",
                        showEmptyGroup: false,
                        collapsed: false,
                        autoPriority: false,
                        items: aTodayNotifs.map(function (n) {
                            return NotificationService.createNotificationListItem(n, function () {
                                that._refreshNotificationPopover();
                            });
                        })
                    });
                    oPopover.addContent(oTodayGroup);
                }

                if (aEarlierNotifs.length > 0) {
                    var oEarlierGroup = new NotificationListGroup({
                        title: "Earlier",
                        showEmptyGroup: false,
                        collapsed: aTodayNotifs.length > 0,
                        autoPriority: false,
                        items: aEarlierNotifs.map(function (n) {
                            return NotificationService.createNotificationListItem(n, function () {
                                that._refreshNotificationPopover();
                            });
                        })
                    });
                    oPopover.addContent(oEarlierGroup);
                }
            }

            this._updateNotificationBadge();
        }
    });
});
