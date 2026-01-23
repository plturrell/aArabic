/* Karma configuration for UI5 app with headless Chrome */
module.exports = function (config) {
  config.set({
    basePath: "",
    frameworks: ["ui5"],
    ui5: {
      type: "application",
      configPath: "ui5.yaml",
      paths: {
        webapp: "."
      }
    },
    browsers: ["ChromeHeadlessNoSandbox"],
    customLaunchers: {
      ChromeHeadlessNoSandbox: {
        base: "ChromeHeadless",
        flags: [
          "--no-sandbox",
          "--disable-gpu",
          "--disable-dev-shm-usage"
        ]
      }
    },
    singleRun: true,
    port: 9876,
    hostname: "127.0.0.1",
    reporters: ["progress"]
  });
};
