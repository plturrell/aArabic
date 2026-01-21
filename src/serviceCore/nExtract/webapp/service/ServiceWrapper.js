sap.ui.define([], function () {
    "use strict";

    // Configuration
    const USE_MOCK = true; 

    // Mock Data
    const MOCK_DOCS = [
        { id: "1", name: "Invoice #1024", date: "Today", icon: "sap-icon://pdf-attachment", confidence: 98, status: "Success" },
        { id: "2", name: "Receipt #55", date: "Yesterday", icon: "sap-icon://camera", confidence: 85, status: "Warning" },
        { id: "3", name: "Contract_v2.docx", date: "Oct 20", icon: "sap-icon://doc-attachment", confidence: 60, status: "Error" }
    ];

    const MOCK_RESULTS = {
        "1": [
            { key: "Invoice Number", value: "INV-1024", confidence: 99 },
            { key: "Total", value: "$500.00", confidence: 98 },
            { key: "Vendor", value: "Acme Corp", confidence: 95 }
        ],
        "2": [
            { key: "Total", value: "$12.50", confidence: 90 },
            { key: "Date", value: "Oct 22", confidence: 85 }
        ],
        "3": [
            { key: "Party A", value: "Unknown", confidence: 45 },
            { key: "Date", value: "2023-??-??", confidence: 60 }
        ]
    };

    return {
        getHistory: function () {
            return new Promise(resolve => {
                setTimeout(() => resolve([...MOCK_DOCS]), 300);
            });
        },

        getDocument: function (sId) {
            return new Promise((resolve, reject) => {
                const oDoc = MOCK_DOCS.find(d => d.id === sId);
                setTimeout(() => {
                    if (oDoc) {
                        resolve({
                            meta: oDoc,
                            results: MOCK_RESULTS[sId] || []
                        });
                    } else {
                        reject("Document not found");
                    }
                }, 200);
            });
        }
    };
});
