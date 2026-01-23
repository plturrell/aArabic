/* global printTestSummary */
(function () {
  if (window.__karma__) {
    // Provide a noop start adapter for our standalone test harness
    if (typeof window.__karma__.start !== "function") {
      window.__karma__.start = function () {};
    }
    window.__karma__.loaded = function () {};

    var finish = function () {
      try {
        var passed = true;
        if (typeof printTestSummary === "function") {
          passed = printTestSummary() !== false;
        }
        window.__karma__.info({ total: 1 });
        window.__karma__.result({
          description: "UIFlow integration suite",
          suite: ["UIFlow"],
          success: passed,
          log: []
        });
        window.__karma__.complete({ coverage: {}, exitCode: passed ? 0 : 1 });
      } catch (e) {
        window.__karma__.error(e);
      }
    };

    if (document.readyState === "complete") {
      finish();
    } else {
      window.addEventListener("load", finish);
    }
  }
})();
