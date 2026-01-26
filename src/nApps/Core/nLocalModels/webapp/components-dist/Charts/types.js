// Charts/types.ts
var CHART_COLORS = {
  primary: "#0a6ed1",
  success: "#107e3e",
  warning: "#e9730c",
  error: "#bb0000",
  neutral: "#6a6d70",
  gradient: {
    blue: ["#0a6ed1", "#1a9fff"],
    green: ["#107e3e", "#2ecc71"],
    orange: ["#e9730c", "#f39c12"],
    red: ["#bb0000", "#e74c3c"]
  },
  series: [
    "#0a6ed1",
    "#107e3e",
    "#e9730c",
    "#9b59b6",
    "#1abc9c",
    "#e74c3c",
    "#3498db",
    "#f39c12"
  ]
};
var DEFAULT_CHART_CONFIG = {
  width: 400,
  height: 300,
  margin: { top: 20, right: 20, bottom: 40, left: 50 },
  animate: true,
  animationDuration: 300,
  responsive: true
};
var DEFAULT_RADIAL_CONFIG = {
  maxValue: 100,
  minValue: 0,
  arcWidth: 20,
  showValue: true,
  showLabel: true,
  thresholds: [
    { value: 33, color: CHART_COLORS.error },
    { value: 66, color: CHART_COLORS.warning },
    { value: 100, color: CHART_COLORS.success }
  ]
};
var DEFAULT_GAUGE_CONFIG = {
  min: 0,
  max: 100,
  showNeedle: true,
  needleColor: "#333",
  zones: [
    { min: 0, max: 33, color: CHART_COLORS.success, label: "Good" },
    { min: 33, max: 66, color: CHART_COLORS.warning, label: "Warning" },
    { min: 66, max: 100, color: CHART_COLORS.error, label: "Critical" }
  ]
};
export {
  DEFAULT_RADIAL_CONFIG,
  DEFAULT_GAUGE_CONFIG,
  DEFAULT_CHART_CONFIG,
  CHART_COLORS
};

//# debugId=31E71209CDB6A8D164756E2164756E21
