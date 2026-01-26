sap.ui.define([
    "sap/m/MessageToast",
    "sap/m/NotificationListItem",
    "sap/m/Button",
    "sap/ui/core/library"
], function (MessageToast, NotificationListItem, Button, coreLibrary) {
    "use strict";

    var Priority = coreLibrary.Priority;

    /**
     * Notification Service for Toast and Persistent Notifications
     * Handles: Toast messages, notification center with history, actions, priorities
     */
    var NotificationService = {

        // Notification type constants
        Types: {
            SUCCESS: "success",
            WARNING: "warning",
            ERROR: "error",
            INFO: "info"
        },

        // Priority levels
        Priority: {
            LOW: "Low",
            MEDIUM: "Medium",
            HIGH: "High"
        },

        // Default configuration
        config: {
            defaultTimeout: 3000,           // 3 seconds for toast
            errorTimeout: 5000,             // 5 seconds for errors
            maxNotifications: 100,          // Maximum stored notifications
            persistToStorage: true          // Persist to localStorage
        },

        // Storage key for persistence
        STORAGE_KEY: "nucleus_notifications",

        // Internal state
        _notifications: [],
        _callbacks: [],
        _nextId: 1,
        _popover: null,

        /**
         * Initialize the notification service
         * @returns {object} This instance for chaining
         */
        init: function () {
            this._loadFromStorage();
            return this;
        },

        /**
         * Show a notification (toast + optional persistent)
         * @param {string} sMessage - The message to display
         * @param {string} sType - Notification type (success, warning, error, info)
         * @param {object} oOptions - Additional options
         * @returns {string} Notification ID
         */
        show: function (sMessage, sType, oOptions) {
            oOptions = oOptions || {};
            sType = sType || this.Types.INFO;

            var sId = "notif_" + this._nextId++;
            var iTimeout = oOptions.timeout !== undefined ? oOptions.timeout : this._getDefaultTimeout(sType);

            // Show toast notification
            if (oOptions.showToast !== false) {
                MessageToast.show(sMessage, {
                    duration: iTimeout,
                    width: oOptions.toastWidth || "15em",
                    closeOnBrowserNavigation: false
                });
            }

            // Add to notification center if persistent
            if (oOptions.persistent !== false) {
                var oNotification = {
                    id: sId,
                    message: sMessage,
                    type: sType,
                    priority: oOptions.priority || this.Priority.MEDIUM,
                    timestamp: new Date().toISOString(),
                    read: false,
                    title: oOptions.title || this._getDefaultTitle(sType),
                    description: oOptions.description || "",
                    actions: oOptions.actions || [],
                    authorName: oOptions.authorName || "System",
                    authorPicture: oOptions.authorPicture || ""
                };

                this._notifications.unshift(oNotification);
                this._trimNotifications();
                this._saveToStorage();
                this._fireNotificationAdded(oNotification);
            }

            return sId;
        },

        /**
         * Show success notification
         * @param {string} sMessage - Message to display
         * @param {object} oOptions - Additional options
         * @returns {string} Notification ID
         */
        showSuccess: function (sMessage, oOptions) {
            return this.show(sMessage, this.Types.SUCCESS, oOptions);
        },

        /**
         * Show error notification
         * @param {string} sMessage - Message to display
         * @param {object} oOptions - Additional options
         * @returns {string} Notification ID
         */
        showError: function (sMessage, oOptions) {
            oOptions = oOptions || {};
            oOptions.priority = oOptions.priority || this.Priority.HIGH;
            return this.show(sMessage, this.Types.ERROR, oOptions);
        },

        /**
         * Show warning notification
         * @param {string} sMessage - Message to display
         * @param {object} oOptions - Additional options
         * @returns {string} Notification ID
         */
        showWarning: function (sMessage, oOptions) {
            return this.show(sMessage, this.Types.WARNING, oOptions);
        },

        /**
         * Show info notification
         * @param {string} sMessage - Message to display
         * @param {object} oOptions - Additional options
         * @returns {string} Notification ID
         */
        showInfo: function (sMessage, oOptions) {
            return this.show(sMessage, this.Types.INFO, oOptions);
        },

        /**
         * Get all notifications
         * @param {boolean} bUnreadOnly - Return only unread notifications
         * @returns {array} Array of notification objects
         */
        getNotifications: function (bUnreadOnly) {
            if (bUnreadOnly) {
                return this._notifications.filter(function (n) { return !n.read; });
            }
            return this._notifications.slice();
        },

        /**
         * Get unread notification count
         * @returns {number} Count of unread notifications
         */
        getUnreadCount: function () {
            return this._notifications.filter(function (n) { return !n.read; }).length;
        },

        /**
         * Mark a notification as read
         * @param {string} sId - Notification ID
         * @returns {boolean} True if notification was found and marked
         */
        markAsRead: function (sId) {
            var oNotification = this._findNotification(sId);
            if (oNotification) {
                oNotification.read = true;
                this._saveToStorage();
                this._fireNotificationChanged(oNotification);
                return true;
            }
            return false;
        },

        /**
         * Mark all notifications as read
         */
        markAllAsRead: function () {
            this._notifications.forEach(function (n) {
                n.read = true;
            });
            this._saveToStorage();
            this._fireNotificationsChanged();
        },

        /**
         * Dismiss (remove) a notification
         * @param {string} sId - Notification ID
         * @returns {boolean} True if notification was found and removed
         */
        dismiss: function (sId) {
            var iIndex = this._findNotificationIndex(sId);
            if (iIndex > -1) {
                var oRemoved = this._notifications.splice(iIndex, 1)[0];
                this._saveToStorage();
                this._fireNotificationRemoved(oRemoved);
                return true;
            }
            return false;
        },

        /**
         * Clear all notifications
         */
        clearAll: function () {
            this._notifications = [];
            this._saveToStorage();
            this._fireNotificationsChanged();
        },

        /**
         * Register callback for notification events
         * @param {string} sEvent - Event name (added, removed, changed, allChanged)
         * @param {function} fnCallback - Callback function
         */
        attachEvent: function (sEvent, fnCallback) {
            if (typeof fnCallback === "function") {
                this._callbacks.push({ event: sEvent, callback: fnCallback });
            }
            return this;
        },

        /**
         * Detach callback for notification events
         * @param {string} sEvent - Event name
         * @param {function} fnCallback - Callback function
         */
        detachEvent: function (sEvent, fnCallback) {
            this._callbacks = this._callbacks.filter(function (cb) {
                return !(cb.event === sEvent && cb.callback === fnCallback);
            });
            return this;
        },

        /**
         * Create a NotificationListItem for UI5 popover
         * @param {object} oNotification - Notification object
         * @param {function} fnOnClose - Close handler
         * @returns {sap.m.NotificationListItem} UI5 notification item
         */
        createNotificationListItem: function (oNotification, fnOnClose) {
            var that = this;
            var aButtons = [];

            // Create action buttons
            if (oNotification.actions && oNotification.actions.length > 0) {
                oNotification.actions.forEach(function (oAction) {
                    aButtons.push(new Button({
                        text: oAction.text,
                        type: oAction.type || "Default",
                        press: function () {
                            if (oAction.handler) {
                                oAction.handler(oNotification);
                            }
                        }
                    }));
                });
            }

            return new NotificationListItem({
                title: oNotification.title,
                description: oNotification.message,
                datetime: this._formatTimestamp(oNotification.timestamp),
                priority: this._mapPriority(oNotification.priority),
                unread: !oNotification.read,
                authorName: oNotification.authorName,
                authorPicture: oNotification.authorPicture,
                showCloseButton: true,
                buttons: aButtons,
                close: function () {
                    that.dismiss(oNotification.id);
                    if (fnOnClose) {
                        fnOnClose(oNotification);
                    }
                },
                press: function () {
                    that.markAsRead(oNotification.id);
                }
            });
        },

        // ========================================================================
        // PRIVATE METHODS
        // ========================================================================

        _getDefaultTimeout: function (sType) {
            if (sType === this.Types.ERROR) {
                return this.config.errorTimeout;
            }
            return this.config.defaultTimeout;
        },

        _getDefaultTitle: function (sType) {
            var mTitles = {
                success: "Success",
                warning: "Warning",
                error: "Error",
                info: "Information"
            };
            return mTitles[sType] || "Notification";
        },

        _mapPriority: function (sPriority) {
            var mPriorityMap = {
                "Low": Priority.Low,
                "Medium": Priority.Medium,
                "High": Priority.High
            };
            return mPriorityMap[sPriority] || Priority.None;
        },

        _findNotification: function (sId) {
            return this._notifications.find(function (n) { return n.id === sId; });
        },

        _findNotificationIndex: function (sId) {
            return this._notifications.findIndex(function (n) { return n.id === sId; });
        },

        _trimNotifications: function () {
            if (this._notifications.length > this.config.maxNotifications) {
                this._notifications = this._notifications.slice(0, this.config.maxNotifications);
            }
        },

        _formatTimestamp: function (sTimestamp) {
            var oDate = new Date(sTimestamp);
            var oNow = new Date();
            var iDiffMs = oNow - oDate;
            var iDiffMins = Math.floor(iDiffMs / 60000);
            var iDiffHours = Math.floor(iDiffMins / 60);
            var iDiffDays = Math.floor(iDiffHours / 24);

            if (iDiffMins < 1) {
                return "Just now";
            } else if (iDiffMins < 60) {
                return iDiffMins + " min ago";
            } else if (iDiffHours < 24) {
                return iDiffHours + " hour" + (iDiffHours > 1 ? "s" : "") + " ago";
            } else if (iDiffDays < 7) {
                return iDiffDays + " day" + (iDiffDays > 1 ? "s" : "") + " ago";
            } else {
                return oDate.toLocaleDateString();
            }
        },

        _loadFromStorage: function () {
            if (!this.config.persistToStorage) {
                return;
            }
            try {
                var sData = window.localStorage.getItem(this.STORAGE_KEY);
                if (sData) {
                    var oData = JSON.parse(sData);
                    this._notifications = oData.notifications || [];
                    this._nextId = oData.nextId || 1;
                }
            } catch (e) {
                console.warn("NotificationService: Could not load from localStorage:", e);
            }
        },

        _saveToStorage: function () {
            if (!this.config.persistToStorage) {
                return;
            }
            try {
                var oData = {
                    notifications: this._notifications,
                    nextId: this._nextId
                };
                window.localStorage.setItem(this.STORAGE_KEY, JSON.stringify(oData));
            } catch (e) {
                console.warn("NotificationService: Could not save to localStorage:", e);
            }
        },

        _fireNotificationAdded: function (oNotification) {
            this._fireEvent("added", oNotification);
        },

        _fireNotificationRemoved: function (oNotification) {
            this._fireEvent("removed", oNotification);
        },

        _fireNotificationChanged: function (oNotification) {
            this._fireEvent("changed", oNotification);
        },

        _fireNotificationsChanged: function () {
            this._fireEvent("allChanged", null);
        },

        _fireEvent: function (sEvent, oData) {
            this._callbacks.forEach(function (cb) {
                if (cb.event === sEvent || cb.event === "all") {
                    try {
                        cb.callback(oData);
                    } catch (e) {
                        console.error("NotificationService: Callback error:", e);
                    }
                }
            });
        }
    };

    // Initialize on load
    NotificationService.init();

    return NotificationService;
});

