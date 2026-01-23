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
        if (typeof printTestSummary === "function") {
          printTestSummary();
        }
        window.__karma__.complete();
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
