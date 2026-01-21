sap.ui.define([], function () {
    "use strict";

    // Configuration
    const USE_MOCK = true; 

    // Mock Data
    const MOCK_PROJECTS = [
        { id: "1", title: "Podcast Ep. 1", date: "Today", icon: "sap-icon://microphone", duration: "45:00", status: "Editing" },
        { id: "2", title: "Ambient Track", date: "Yesterday", icon: "sap-icon://sound-loud", duration: "03:20", status: "Ready" },
        { id: "3", title: "Voiceover Demo", date: "Jan 15", icon: "sap-icon://marketing-campaign", duration: "01:15", status: "Draft" }
    ];

    return {
        getProjects: function () {
            return new Promise(resolve => {
                setTimeout(() => resolve([...MOCK_PROJECTS]), 300);
            });
        },

        getProject: function (sId) {
            return new Promise((resolve, reject) => {
                const oProj = MOCK_PROJECTS.find(p => p.id === sId);
                setTimeout(() => {
                    if (oProj) resolve(oProj);
                    else reject("Project not found");
                }, 200);
            });
        }
    };
});
