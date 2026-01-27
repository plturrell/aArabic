/* Karma configuration running the standalone integration tests with headless Chrome */
module.exports = function (config) {
  config.set({
    basePath: "",
    frameworks: [],
    files: [
      { pattern: "test/integration/UIFlow.test.js", watched: false },
      { pattern: "test/karma.bootstrap.js", watched: false }
    ],
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
