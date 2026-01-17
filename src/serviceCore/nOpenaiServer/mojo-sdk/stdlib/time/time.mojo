"""
Mojo Time Module - Date/time handling and duration utilities.

This module provides comprehensive time handling including:
- Duration for time intervals
- Date for calendar dates
- Time for time of day
- DateTime combining date and time
- Timezone support
- Formatting and parsing
- Timer utilities
"""

# ============================================================================
# Time Constants
# ============================================================================

alias NANOS_PER_MICRO: Int64 = 1000
alias NANOS_PER_MILLI: Int64 = 1000000
alias NANOS_PER_SEC: Int64 = 1000000000
alias NANOS_PER_MIN: Int64 = 60 * NANOS_PER_SEC
alias NANOS_PER_HOUR: Int64 = 60 * NANOS_PER_MIN
alias NANOS_PER_DAY: Int64 = 24 * NANOS_PER_HOUR

alias MICROS_PER_MILLI: Int64 = 1000
alias MICROS_PER_SEC: Int64 = 1000000
alias MILLIS_PER_SEC: Int64 = 1000

alias SECS_PER_MIN: Int64 = 60
alias SECS_PER_HOUR: Int64 = 3600
alias SECS_PER_DAY: Int64 = 86400

alias DAYS_PER_WEEK: Int = 7
alias MONTHS_PER_YEAR: Int = 12

# Days in each month (non-leap year)
alias DAYS_IN_MONTH: List[Int] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Unix epoch: January 1, 1970
alias UNIX_EPOCH_YEAR: Int = 1970
alias UNIX_EPOCH_MONTH: Int = 1
alias UNIX_EPOCH_DAY: Int = 1

# ============================================================================
# Weekday Enumeration
# ============================================================================

struct Weekday:
    """Represents a day of the week."""
    var value: Int

    alias MONDAY = Weekday(0)
    alias TUESDAY = Weekday(1)
    alias WEDNESDAY = Weekday(2)
    alias THURSDAY = Weekday(3)
    alias FRIDAY = Weekday(4)
    alias SATURDAY = Weekday(5)
    alias SUNDAY = Weekday(6)

    fn __init__(inout self, value: Int):
        self.value = value % 7

    fn name(self) -> String:
        """Returns the full name of the weekday."""
        if self.value == 0:
            return "Monday"
        elif self.value == 1:
            return "Tuesday"
        elif self.value == 2:
            return "Wednesday"
        elif self.value == 3:
            return "Thursday"
        elif self.value == 4:
            return "Friday"
        elif self.value == 5:
            return "Saturday"
        else:
            return "Sunday"

    fn short_name(self) -> String:
        """Returns the abbreviated name (Mon, Tue, etc.)."""
        if self.value == 0:
            return "Mon"
        elif self.value == 1:
            return "Tue"
        elif self.value == 2:
            return "Wed"
        elif self.value == 3:
            return "Thu"
        elif self.value == 4:
            return "Fri"
        elif self.value == 5:
            return "Sat"
        else:
            return "Sun"

    fn is_weekend(self) -> Bool:
        """Returns True if Saturday or Sunday."""
        return self.value >= 5

    fn is_weekday(self) -> Bool:
        """Returns True if Monday through Friday."""
        return self.value < 5

# ============================================================================
# Month Enumeration
# ============================================================================

struct Month:
    """Represents a month of the year."""
    var value: Int  # 1-12

    alias JANUARY = Month(1)
    alias FEBRUARY = Month(2)
    alias MARCH = Month(3)
    alias APRIL = Month(4)
    alias MAY = Month(5)
    alias JUNE = Month(6)
    alias JULY = Month(7)
    alias AUGUST = Month(8)
    alias SEPTEMBER = Month(9)
    alias OCTOBER = Month(10)
    alias NOVEMBER = Month(11)
    alias DECEMBER = Month(12)

    fn __init__(inout self, value: Int):
        # Clamp to valid range
        if value < 1:
            self.value = 1
        elif value > 12:
            self.value = 12
        else:
            self.value = value

    fn name(self) -> String:
        """Returns the full month name."""
        if self.value == 1:
            return "January"
        elif self.value == 2:
            return "February"
        elif self.value == 3:
            return "March"
        elif self.value == 4:
            return "April"
        elif self.value == 5:
            return "May"
        elif self.value == 6:
            return "June"
        elif self.value == 7:
            return "July"
        elif self.value == 8:
            return "August"
        elif self.value == 9:
            return "September"
        elif self.value == 10:
            return "October"
        elif self.value == 11:
            return "November"
        else:
            return "December"

    fn short_name(self) -> String:
        """Returns the abbreviated month name."""
        if self.value == 1:
            return "Jan"
        elif self.value == 2:
            return "Feb"
        elif self.value == 3:
            return "Mar"
        elif self.value == 4:
            return "Apr"
        elif self.value == 5:
            return "May"
        elif self.value == 6:
            return "Jun"
        elif self.value == 7:
            return "Jul"
        elif self.value == 8:
            return "Aug"
        elif self.value == 9:
            return "Sep"
        elif self.value == 10:
            return "Oct"
        elif self.value == 11:
            return "Nov"
        else:
            return "Dec"

    fn days(self, is_leap_year: Bool = False) -> Int:
        """Returns the number of days in this month."""
        if self.value == 2 and is_leap_year:
            return 29
        return DAYS_IN_MONTH[self.value - 1]

# ============================================================================
# Duration - Time Interval
# ============================================================================

struct Duration:
    """
    Represents a span of time with nanosecond precision.

    Duration can be positive or negative, allowing for time arithmetic.
    """
    var nanos: Int64  # Total nanoseconds

    # Zero duration
    alias ZERO = Duration(0)

    # Common durations
    alias NANOSECOND = Duration(1)
    alias MICROSECOND = Duration(NANOS_PER_MICRO)
    alias MILLISECOND = Duration(NANOS_PER_MILLI)
    alias SECOND = Duration(NANOS_PER_SEC)
    alias MINUTE = Duration(NANOS_PER_MIN)
    alias HOUR = Duration(NANOS_PER_HOUR)
    alias DAY = Duration(NANOS_PER_DAY)

    fn __init__(inout self, nanos: Int64):
        """Creates a duration from nanoseconds."""
        self.nanos = nanos

    @staticmethod
    fn from_nanos(nanos: Int64) -> Duration:
        """Creates a duration from nanoseconds."""
        return Duration(nanos)

    @staticmethod
    fn from_micros(micros: Int64) -> Duration:
        """Creates a duration from microseconds."""
        return Duration(micros * NANOS_PER_MICRO)

    @staticmethod
    fn from_millis(millis: Int64) -> Duration:
        """Creates a duration from milliseconds."""
        return Duration(millis * NANOS_PER_MILLI)

    @staticmethod
    fn from_secs(secs: Int64) -> Duration:
        """Creates a duration from seconds."""
        return Duration(secs * NANOS_PER_SEC)

    @staticmethod
    fn from_mins(mins: Int64) -> Duration:
        """Creates a duration from minutes."""
        return Duration(mins * NANOS_PER_MIN)

    @staticmethod
    fn from_hours(hours: Int64) -> Duration:
        """Creates a duration from hours."""
        return Duration(hours * NANOS_PER_HOUR)

    @staticmethod
    fn from_days(days: Int64) -> Duration:
        """Creates a duration from days."""
        return Duration(days * NANOS_PER_DAY)

    @staticmethod
    fn from_hms(hours: Int, mins: Int, secs: Int) -> Duration:
        """Creates a duration from hours, minutes, seconds."""
        let total_secs = hours * 3600 + mins * 60 + secs
        return Duration(Int64(total_secs) * NANOS_PER_SEC)

    # Getters
    fn total_nanos(self) -> Int64:
        """Returns total nanoseconds."""
        return self.nanos

    fn total_micros(self) -> Int64:
        """Returns total microseconds."""
        return self.nanos // NANOS_PER_MICRO

    fn total_millis(self) -> Int64:
        """Returns total milliseconds."""
        return self.nanos // NANOS_PER_MILLI

    fn total_secs(self) -> Int64:
        """Returns total seconds."""
        return self.nanos // NANOS_PER_SEC

    fn total_mins(self) -> Int64:
        """Returns total minutes."""
        return self.nanos // NANOS_PER_MIN

    fn total_hours(self) -> Int64:
        """Returns total hours."""
        return self.nanos // NANOS_PER_HOUR

    fn total_days(self) -> Int64:
        """Returns total days."""
        return self.nanos // NANOS_PER_DAY

    fn as_secs_f64(self) -> Float64:
        """Returns duration as floating-point seconds."""
        return Float64(self.nanos) / Float64(NANOS_PER_SEC)

    # Component extraction
    fn subsec_nanos(self) -> Int:
        """Returns nanosecond component (0-999,999,999)."""
        return Int(self.nanos % NANOS_PER_SEC)

    fn subsec_micros(self) -> Int:
        """Returns microsecond component (0-999,999)."""
        return Int((self.nanos % NANOS_PER_SEC) // NANOS_PER_MICRO)

    fn subsec_millis(self) -> Int:
        """Returns millisecond component (0-999)."""
        return Int((self.nanos % NANOS_PER_SEC) // NANOS_PER_MILLI)

    # Properties
    fn is_zero(self) -> Bool:
        """Returns True if duration is zero."""
        return self.nanos == 0

    fn is_positive(self) -> Bool:
        """Returns True if duration is positive."""
        return self.nanos > 0

    fn is_negative(self) -> Bool:
        """Returns True if duration is negative."""
        return self.nanos < 0

    fn abs(self) -> Duration:
        """Returns absolute value of duration."""
        if self.nanos < 0:
            return Duration(-self.nanos)
        return Duration(self.nanos)

    # Arithmetic
    fn __add__(self, other: Duration) -> Duration:
        """Adds two durations."""
        return Duration(self.nanos + other.nanos)

    fn __sub__(self, other: Duration) -> Duration:
        """Subtracts two durations."""
        return Duration(self.nanos - other.nanos)

    fn __mul__(self, scalar: Int64) -> Duration:
        """Multiplies duration by scalar."""
        return Duration(self.nanos * scalar)

    fn __truediv__(self, divisor: Int64) -> Duration:
        """Divides duration by scalar."""
        return Duration(self.nanos // divisor)

    fn __neg__(self) -> Duration:
        """Negates duration."""
        return Duration(-self.nanos)

    # Comparison
    fn __eq__(self, other: Duration) -> Bool:
        return self.nanos == other.nanos

    fn __ne__(self, other: Duration) -> Bool:
        return self.nanos != other.nanos

    fn __lt__(self, other: Duration) -> Bool:
        return self.nanos < other.nanos

    fn __le__(self, other: Duration) -> Bool:
        return self.nanos <= other.nanos

    fn __gt__(self, other: Duration) -> Bool:
        return self.nanos > other.nanos

    fn __ge__(self, other: Duration) -> Bool:
        return self.nanos >= other.nanos

    fn to_string(self) -> String:
        """Formats duration as human-readable string."""
        var result = String("")
        var remaining = self.nanos

        if remaining < 0:
            result = "-"
            remaining = -remaining

        let days = remaining // NANOS_PER_DAY
        remaining = remaining % NANOS_PER_DAY

        let hours = remaining // NANOS_PER_HOUR
        remaining = remaining % NANOS_PER_HOUR

        let mins = remaining // NANOS_PER_MIN
        remaining = remaining % NANOS_PER_MIN

        let secs = remaining // NANOS_PER_SEC
        remaining = remaining % NANOS_PER_SEC

        if days > 0:
            result = result + String(days) + "d "
        if hours > 0 or days > 0:
            result = result + String(hours) + "h "
        if mins > 0 or hours > 0 or days > 0:
            result = result + String(mins) + "m "

        result = result + String(secs)

        if remaining > 0:
            result = result + "." + String(remaining // NANOS_PER_MILLI)

        result = result + "s"
        return result

# ============================================================================
# Date - Calendar Date
# ============================================================================

struct Date:
    """
    Represents a calendar date (year, month, day).

    Uses the proleptic Gregorian calendar.
    """
    var year: Int
    var month: Int   # 1-12
    var day: Int     # 1-31

    fn __init__(inout self, year: Int, month: Int, day: Int):
        """Creates a date from year, month, day."""
        self.year = year
        self.month = month
        self.day = day

    @staticmethod
    fn today() -> Date:
        """Returns the current date (placeholder - needs system time)."""
        # This would need system time integration
        return Date(2026, 1, 15)

    @staticmethod
    fn from_ordinal(year: Int, day_of_year: Int) -> Date:
        """Creates a date from year and day of year (1-366)."""
        let is_leap = Date._is_leap_year(year)
        var remaining = day_of_year
        var month = 1

        while month <= 12:
            var days_in_month = DAYS_IN_MONTH[month - 1]
            if month == 2 and is_leap:
                days_in_month = 29

            if remaining <= days_in_month:
                return Date(year, month, remaining)

            remaining -= days_in_month
            month += 1

        # Default to last day of year
        return Date(year, 12, 31)

    @staticmethod
    fn from_timestamp(timestamp: Int64) -> Date:
        """Creates a date from Unix timestamp (seconds since epoch)."""
        let days_since_epoch = timestamp // SECS_PER_DAY
        return Date._from_days_since_epoch(Int(days_since_epoch))

    @staticmethod
    fn _from_days_since_epoch(days: Int) -> Date:
        """Converts days since Unix epoch to date."""
        var remaining_days = days
        var year = UNIX_EPOCH_YEAR

        # Handle negative days (before epoch)
        while remaining_days < 0:
            year -= 1
            remaining_days += Date._days_in_year(year)

        # Find the year
        while remaining_days >= Date._days_in_year(year):
            remaining_days -= Date._days_in_year(year)
            year += 1

        # Find month and day
        let is_leap = Date._is_leap_year(year)
        var month = 1

        while month <= 12:
            var days_in_month = DAYS_IN_MONTH[month - 1]
            if month == 2 and is_leap:
                days_in_month = 29

            if remaining_days < days_in_month:
                return Date(year, month, remaining_days + 1)

            remaining_days -= days_in_month
            month += 1

        return Date(year, 12, 31)

    @staticmethod
    fn _is_leap_year(year: Int) -> Bool:
        """Returns True if year is a leap year."""
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    @staticmethod
    fn _days_in_year(year: Int) -> Int:
        """Returns days in a given year."""
        if Date._is_leap_year(year):
            return 366
        return 365

    fn is_leap_year(self) -> Bool:
        """Returns True if this date's year is a leap year."""
        return Date._is_leap_year(self.year)

    fn days_in_month(self) -> Int:
        """Returns the number of days in this date's month."""
        if self.month == 2 and self.is_leap_year():
            return 29
        return DAYS_IN_MONTH[self.month - 1]

    fn days_in_year(self) -> Int:
        """Returns days in this date's year."""
        return Date._days_in_year(self.year)

    fn day_of_year(self) -> Int:
        """Returns the day of the year (1-366)."""
        var day_num = 0
        let is_leap = self.is_leap_year()

        for m in range(1, self.month):
            if m == 2 and is_leap:
                day_num += 29
            else:
                day_num += DAYS_IN_MONTH[m - 1]

        return day_num + self.day

    fn weekday(self) -> Weekday:
        """Returns the day of the week."""
        # Using Zeller's congruence (modified for Monday=0)
        var y = self.year
        var m = self.month

        if m < 3:
            m += 12
            y -= 1

        let k = y % 100
        let j = y // 100

        var h = (self.day + (13 * (m + 1)) // 5 + k + k // 4 + j // 4 - 2 * j) % 7

        # Convert from Zeller (0=Saturday) to Monday=0
        h = (h + 5) % 7

        return Weekday(h)

    fn week_of_year(self) -> Int:
        """Returns ISO week number (1-53)."""
        # Find day of year for this date
        let day_num = self.day_of_year()

        # Find weekday of Jan 1
        let jan1 = Date(self.year, 1, 1)
        let jan1_weekday = jan1.weekday().value

        # Calculate week number
        return (day_num + jan1_weekday - 1) // 7 + 1

    fn quarter(self) -> Int:
        """Returns the quarter (1-4)."""
        return (self.month - 1) // 3 + 1

    fn get_month(self) -> Month:
        """Returns the month as Month struct."""
        return Month(self.month)

    fn is_valid(self) -> Bool:
        """Returns True if this is a valid date."""
        if self.month < 1 or self.month > 12:
            return False
        if self.day < 1:
            return False
        if self.day > self.days_in_month():
            return False
        return True

    # Date arithmetic
    fn add_days(self, days: Int) -> Date:
        """Returns a new date with days added."""
        let total_days = self._to_days_since_epoch() + days
        return Date._from_days_since_epoch(total_days)

    fn add_months(self, months: Int) -> Date:
        """Returns a new date with months added."""
        var new_month = self.month + months
        var new_year = self.year

        while new_month > 12:
            new_month -= 12
            new_year += 1

        while new_month < 1:
            new_month += 12
            new_year -= 1

        # Clamp day to valid range for new month
        let temp = Date(new_year, new_month, 1)
        let max_day = temp.days_in_month()
        let new_day = self.day if self.day <= max_day else max_day

        return Date(new_year, new_month, new_day)

    fn add_years(self, years: Int) -> Date:
        """Returns a new date with years added."""
        let new_year = self.year + years

        # Handle Feb 29 in non-leap year
        var new_day = self.day
        if self.month == 2 and self.day == 29:
            if not Date._is_leap_year(new_year):
                new_day = 28

        return Date(new_year, self.month, new_day)

    fn _to_days_since_epoch(self) -> Int:
        """Converts date to days since Unix epoch."""
        var days = 0

        # Add days for complete years
        if self.year >= UNIX_EPOCH_YEAR:
            for y in range(UNIX_EPOCH_YEAR, self.year):
                days += Date._days_in_year(y)
        else:
            for y in range(self.year, UNIX_EPOCH_YEAR):
                days -= Date._days_in_year(y)

        # Add days for complete months in current year
        let is_leap = self.is_leap_year()
        for m in range(1, self.month):
            if m == 2 and is_leap:
                days += 29
            else:
                days += DAYS_IN_MONTH[m - 1]

        # Add remaining days
        days += self.day - 1

        return days

    fn days_until(self, other: Date) -> Int:
        """Returns days between this date and another."""
        return other._to_days_since_epoch() - self._to_days_since_epoch()

    # Comparison
    fn __eq__(self, other: Date) -> Bool:
        return self.year == other.year and self.month == other.month and self.day == other.day

    fn __ne__(self, other: Date) -> Bool:
        return not self.__eq__(other)

    fn __lt__(self, other: Date) -> Bool:
        if self.year != other.year:
            return self.year < other.year
        if self.month != other.month:
            return self.month < other.month
        return self.day < other.day

    fn __le__(self, other: Date) -> Bool:
        return self.__lt__(other) or self.__eq__(other)

    fn __gt__(self, other: Date) -> Bool:
        return not self.__le__(other)

    fn __ge__(self, other: Date) -> Bool:
        return not self.__lt__(other)

    # Formatting
    fn to_iso_string(self) -> String:
        """Formats as ISO 8601 date (YYYY-MM-DD)."""
        var result = String(self.year)

        result = result + "-"
        if self.month < 10:
            result = result + "0"
        result = result + String(self.month)

        result = result + "-"
        if self.day < 10:
            result = result + "0"
        result = result + String(self.day)

        return result

    fn to_string(self) -> String:
        """Formats as human-readable date."""
        let month = Month(self.month)
        return month.name() + " " + String(self.day) + ", " + String(self.year)

    fn format(self, pattern: String) -> String:
        """
        Formats date according to pattern.

        Supported patterns:
        - %Y: 4-digit year
        - %m: 2-digit month (01-12)
        - %d: 2-digit day (01-31)
        - %B: Full month name
        - %b: Abbreviated month name
        - %A: Full weekday name
        - %a: Abbreviated weekday name
        - %j: Day of year (001-366)
        - %W: Week number (01-53)
        """
        var result = String("")
        var i = 0

        while i < len(pattern):
            if pattern[i] == "%" and i + 1 < len(pattern):
                let code = pattern[i + 1]

                if code == "Y":
                    result = result + String(self.year)
                elif code == "m":
                    if self.month < 10:
                        result = result + "0"
                    result = result + String(self.month)
                elif code == "d":
                    if self.day < 10:
                        result = result + "0"
                    result = result + String(self.day)
                elif code == "B":
                    result = result + Month(self.month).name()
                elif code == "b":
                    result = result + Month(self.month).short_name()
                elif code == "A":
                    result = result + self.weekday().name()
                elif code == "a":
                    result = result + self.weekday().short_name()
                elif code == "j":
                    let doy = self.day_of_year()
                    if doy < 10:
                        result = result + "00"
                    elif doy < 100:
                        result = result + "0"
                    result = result + String(doy)
                elif code == "W":
                    let week = self.week_of_year()
                    if week < 10:
                        result = result + "0"
                    result = result + String(week)
                elif code == "%":
                    result = result + "%"
                else:
                    result = result + "%" + code

                i += 2
            else:
                result = result + pattern[i]
                i += 1

        return result

# ============================================================================
# Time - Time of Day
# ============================================================================

struct Time:
    """
    Represents a time of day with nanosecond precision.
    """
    var hour: Int      # 0-23
    var minute: Int    # 0-59
    var second: Int    # 0-59
    var nanosecond: Int  # 0-999,999,999

    fn __init__(inout self, hour: Int, minute: Int, second: Int = 0, nanosecond: Int = 0):
        """Creates a time from components."""
        self.hour = hour % 24
        self.minute = minute % 60
        self.second = second % 60
        self.nanosecond = nanosecond % 1000000000

    @staticmethod
    fn midnight() -> Time:
        """Returns midnight (00:00:00)."""
        return Time(0, 0, 0, 0)

    @staticmethod
    fn noon() -> Time:
        """Returns noon (12:00:00)."""
        return Time(12, 0, 0, 0)

    @staticmethod
    fn from_secs_since_midnight(secs: Int) -> Time:
        """Creates time from seconds since midnight."""
        let hours = secs // 3600
        let mins = (secs % 3600) // 60
        let seconds = secs % 60
        return Time(hours, mins, seconds)

    @staticmethod
    fn from_nanos_since_midnight(nanos: Int64) -> Time:
        """Creates time from nanoseconds since midnight."""
        let total_secs = nanos // NANOS_PER_SEC
        let nano_part = Int(nanos % NANOS_PER_SEC)

        let hours = Int(total_secs // 3600)
        let mins = Int((total_secs % 3600) // 60)
        let seconds = Int(total_secs % 60)

        return Time(hours, mins, seconds, nano_part)

    fn to_secs_since_midnight(self) -> Int:
        """Returns seconds since midnight."""
        return self.hour * 3600 + self.minute * 60 + self.second

    fn to_nanos_since_midnight(self) -> Int64:
        """Returns nanoseconds since midnight."""
        let secs = Int64(self.to_secs_since_midnight())
        return secs * NANOS_PER_SEC + Int64(self.nanosecond)

    fn is_valid(self) -> Bool:
        """Returns True if this is a valid time."""
        return (self.hour >= 0 and self.hour < 24 and
                self.minute >= 0 and self.minute < 60 and
                self.second >= 0 and self.second < 60 and
                self.nanosecond >= 0 and self.nanosecond < 1000000000)

    fn is_am(self) -> Bool:
        """Returns True if before noon."""
        return self.hour < 12

    fn is_pm(self) -> Bool:
        """Returns True if noon or later."""
        return self.hour >= 12

    fn hour_12(self) -> Int:
        """Returns hour in 12-hour format (1-12)."""
        if self.hour == 0:
            return 12
        elif self.hour > 12:
            return self.hour - 12
        return self.hour

    # Arithmetic
    fn add(self, duration: Duration) -> Time:
        """Adds duration to time (wraps at midnight)."""
        let total_nanos = self.to_nanos_since_midnight() + duration.nanos
        let wrapped = total_nanos % NANOS_PER_DAY
        let positive = wrapped if wrapped >= 0 else wrapped + NANOS_PER_DAY
        return Time.from_nanos_since_midnight(positive)

    fn subtract(self, duration: Duration) -> Time:
        """Subtracts duration from time (wraps at midnight)."""
        return self.add(-duration)

    fn duration_until(self, other: Time) -> Duration:
        """Returns duration from this time to another."""
        let diff = other.to_nanos_since_midnight() - self.to_nanos_since_midnight()
        return Duration(diff)

    # Comparison
    fn __eq__(self, other: Time) -> Bool:
        return (self.hour == other.hour and self.minute == other.minute and
                self.second == other.second and self.nanosecond == other.nanosecond)

    fn __ne__(self, other: Time) -> Bool:
        return not self.__eq__(other)

    fn __lt__(self, other: Time) -> Bool:
        if self.hour != other.hour:
            return self.hour < other.hour
        if self.minute != other.minute:
            return self.minute < other.minute
        if self.second != other.second:
            return self.second < other.second
        return self.nanosecond < other.nanosecond

    fn __le__(self, other: Time) -> Bool:
        return self.__lt__(other) or self.__eq__(other)

    fn __gt__(self, other: Time) -> Bool:
        return not self.__le__(other)

    fn __ge__(self, other: Time) -> Bool:
        return not self.__lt__(other)

    # Formatting
    fn to_iso_string(self) -> String:
        """Formats as ISO 8601 time (HH:MM:SS)."""
        var result = String("")

        if self.hour < 10:
            result = result + "0"
        result = result + String(self.hour) + ":"

        if self.minute < 10:
            result = result + "0"
        result = result + String(self.minute) + ":"

        if self.second < 10:
            result = result + "0"
        result = result + String(self.second)

        return result

    fn to_string_12h(self) -> String:
        """Formats as 12-hour time with AM/PM."""
        var result = String("")

        let h = self.hour_12()
        if h < 10:
            result = result + " "
        result = result + String(h) + ":"

        if self.minute < 10:
            result = result + "0"
        result = result + String(self.minute)

        if self.is_am():
            result = result + " AM"
        else:
            result = result + " PM"

        return result

    fn format(self, pattern: String) -> String:
        """
        Formats time according to pattern.

        Supported patterns:
        - %H: 24-hour (00-23)
        - %I: 12-hour (01-12)
        - %M: Minutes (00-59)
        - %S: Seconds (00-59)
        - %f: Microseconds (000000-999999)
        - %p: AM/PM
        """
        var result = String("")
        var i = 0

        while i < len(pattern):
            if pattern[i] == "%" and i + 1 < len(pattern):
                let code = pattern[i + 1]

                if code == "H":
                    if self.hour < 10:
                        result = result + "0"
                    result = result + String(self.hour)
                elif code == "I":
                    let h = self.hour_12()
                    if h < 10:
                        result = result + "0"
                    result = result + String(h)
                elif code == "M":
                    if self.minute < 10:
                        result = result + "0"
                    result = result + String(self.minute)
                elif code == "S":
                    if self.second < 10:
                        result = result + "0"
                    result = result + String(self.second)
                elif code == "f":
                    let micros = self.nanosecond // 1000
                    if micros < 10:
                        result = result + "00000"
                    elif micros < 100:
                        result = result + "0000"
                    elif micros < 1000:
                        result = result + "000"
                    elif micros < 10000:
                        result = result + "00"
                    elif micros < 100000:
                        result = result + "0"
                    result = result + String(micros)
                elif code == "p":
                    if self.is_am():
                        result = result + "AM"
                    else:
                        result = result + "PM"
                elif code == "%":
                    result = result + "%"
                else:
                    result = result + "%" + code

                i += 2
            else:
                result = result + pattern[i]
                i += 1

        return result

# ============================================================================
# DateTime - Combined Date and Time
# ============================================================================

struct DateTime:
    """
    Represents a combined date and time with nanosecond precision.
    """
    var date: Date
    var time: Time

    fn __init__(inout self, date: Date, time: Time):
        """Creates DateTime from Date and Time."""
        self.date = date
        self.time = time

    fn __init__(inout self, year: Int, month: Int, day: Int,
                hour: Int = 0, minute: Int = 0, second: Int = 0, nanosecond: Int = 0):
        """Creates DateTime from components."""
        self.date = Date(year, month, day)
        self.time = Time(hour, minute, second, nanosecond)

    @staticmethod
    fn now() -> DateTime:
        """Returns current date and time (placeholder - needs system time)."""
        # Would need system time integration
        return DateTime(2026, 1, 15, 12, 0, 0)

    @staticmethod
    fn from_timestamp(timestamp: Int64) -> DateTime:
        """Creates DateTime from Unix timestamp (seconds since epoch)."""
        let date = Date.from_timestamp(timestamp)
        let secs_in_day = Int(timestamp % SECS_PER_DAY)
        let time = Time.from_secs_since_midnight(secs_in_day)
        return DateTime(date, time)

    @staticmethod
    fn from_timestamp_millis(timestamp_ms: Int64) -> DateTime:
        """Creates DateTime from Unix timestamp in milliseconds."""
        let timestamp = timestamp_ms // 1000
        let date = Date.from_timestamp(timestamp)
        let secs_in_day = Int(timestamp % SECS_PER_DAY)
        let millis = Int(timestamp_ms % 1000)
        let time = Time.from_secs_since_midnight(secs_in_day)
        return DateTime(date, Time(time.hour, time.minute, time.second, millis * 1000000))

    fn to_timestamp(self) -> Int64:
        """Converts to Unix timestamp (seconds since epoch)."""
        let days = Int64(self.date._to_days_since_epoch())
        let secs = Int64(self.time.to_secs_since_midnight())
        return days * SECS_PER_DAY + secs

    fn to_timestamp_millis(self) -> Int64:
        """Converts to Unix timestamp in milliseconds."""
        return self.to_timestamp() * 1000 + Int64(self.time.nanosecond // 1000000)

    # Component access
    fn year(self) -> Int:
        return self.date.year

    fn month(self) -> Int:
        return self.date.month

    fn day(self) -> Int:
        return self.date.day

    fn hour(self) -> Int:
        return self.time.hour

    fn minute(self) -> Int:
        return self.time.minute

    fn second(self) -> Int:
        return self.time.second

    fn nanosecond(self) -> Int:
        return self.time.nanosecond

    fn weekday(self) -> Weekday:
        return self.date.weekday()

    fn day_of_year(self) -> Int:
        return self.date.day_of_year()

    fn is_valid(self) -> Bool:
        """Returns True if date and time are valid."""
        return self.date.is_valid() and self.time.is_valid()

    # Arithmetic
    fn add(self, duration: Duration) -> DateTime:
        """Adds duration to datetime."""
        let total_nanos = self.time.to_nanos_since_midnight() + duration.nanos

        var extra_days = Int(total_nanos // NANOS_PER_DAY)
        var remaining_nanos = total_nanos % NANOS_PER_DAY

        if remaining_nanos < 0:
            extra_days -= 1
            remaining_nanos += NANOS_PER_DAY

        let new_date = self.date.add_days(extra_days)
        let new_time = Time.from_nanos_since_midnight(remaining_nanos)

        return DateTime(new_date, new_time)

    fn subtract(self, duration: Duration) -> DateTime:
        """Subtracts duration from datetime."""
        return self.add(-duration)

    fn add_days(self, days: Int) -> DateTime:
        """Returns DateTime with days added."""
        return DateTime(self.date.add_days(days), self.time)

    fn add_months(self, months: Int) -> DateTime:
        """Returns DateTime with months added."""
        return DateTime(self.date.add_months(months), self.time)

    fn add_years(self, years: Int) -> DateTime:
        """Returns DateTime with years added."""
        return DateTime(self.date.add_years(years), self.time)

    fn duration_since(self, other: DateTime) -> Duration:
        """Returns duration since another datetime."""
        let self_ts = self.to_timestamp() * NANOS_PER_SEC + Int64(self.time.nanosecond)
        let other_ts = other.to_timestamp() * NANOS_PER_SEC + Int64(other.time.nanosecond)
        return Duration(self_ts - other_ts)

    # Comparison
    fn __eq__(self, other: DateTime) -> Bool:
        return self.date == other.date and self.time == other.time

    fn __ne__(self, other: DateTime) -> Bool:
        return not self.__eq__(other)

    fn __lt__(self, other: DateTime) -> Bool:
        if self.date != other.date:
            return self.date < other.date
        return self.time < other.time

    fn __le__(self, other: DateTime) -> Bool:
        return self.__lt__(other) or self.__eq__(other)

    fn __gt__(self, other: DateTime) -> Bool:
        return not self.__le__(other)

    fn __ge__(self, other: DateTime) -> Bool:
        return not self.__lt__(other)

    # Formatting
    fn to_iso_string(self) -> String:
        """Formats as ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS)."""
        return self.date.to_iso_string() + "T" + self.time.to_iso_string()

    fn to_string(self) -> String:
        """Formats as human-readable datetime."""
        return self.date.to_string() + " " + self.time.to_iso_string()

    fn format(self, pattern: String) -> String:
        """Formats datetime according to pattern (combines Date and Time patterns)."""
        # First apply date formatting, then time formatting
        let date_formatted = self.date.format(pattern)
        return self.time.format(date_formatted)

# ============================================================================
# Timezone
# ============================================================================

struct Timezone:
    """
    Represents a timezone offset from UTC.
    """
    var offset_minutes: Int  # Minutes from UTC (-720 to +840)
    var name: String

    # Common timezones
    alias UTC = Timezone(0, "UTC")
    alias EST = Timezone(-300, "EST")  # -5:00
    alias EDT = Timezone(-240, "EDT")  # -4:00
    alias CST = Timezone(-360, "CST")  # -6:00
    alias CDT = Timezone(-300, "CDT")  # -5:00
    alias MST = Timezone(-420, "MST")  # -7:00
    alias MDT = Timezone(-360, "MDT")  # -6:00
    alias PST = Timezone(-480, "PST")  # -8:00
    alias PDT = Timezone(-420, "PDT")  # -7:00
    alias GMT = Timezone(0, "GMT")
    alias CET = Timezone(60, "CET")    # +1:00
    alias CEST = Timezone(120, "CEST") # +2:00
    alias JST = Timezone(540, "JST")   # +9:00
    alias AST = Timezone(180, "AST")   # Arabia Standard Time +3:00

    fn __init__(inout self, offset_minutes: Int, name: String = ""):
        """Creates timezone from offset in minutes."""
        self.offset_minutes = offset_minutes
        self.name = name

    @staticmethod
    fn from_hours(hours: Int) -> Timezone:
        """Creates timezone from hour offset."""
        return Timezone(hours * 60)

    @staticmethod
    fn from_hm(hours: Int, minutes: Int) -> Timezone:
        """Creates timezone from hours and minutes."""
        let sign = 1 if hours >= 0 else -1
        return Timezone(hours * 60 + sign * minutes)

    fn offset_duration(self) -> Duration:
        """Returns offset as Duration."""
        return Duration.from_mins(Int64(self.offset_minutes))

    fn offset_string(self) -> String:
        """Returns offset as +HH:MM or -HH:MM string."""
        var result = String("")

        let abs_offset = self.offset_minutes if self.offset_minutes >= 0 else -self.offset_minutes
        let hours = abs_offset // 60
        let mins = abs_offset % 60

        if self.offset_minutes >= 0:
            result = "+"
        else:
            result = "-"

        if hours < 10:
            result = result + "0"
        result = result + String(hours) + ":"

        if mins < 10:
            result = result + "0"
        result = result + String(mins)

        return result

    fn to_string(self) -> String:
        """Returns timezone string representation."""
        if len(self.name) > 0:
            return self.name + " (" + self.offset_string() + ")"
        return self.offset_string()

# ============================================================================
# Timer - Stopwatch/Elapsed Time
# ============================================================================

struct Timer:
    """
    Simple timer for measuring elapsed time.

    Note: Actual time measurement requires system clock integration.
    This provides the structure for timing operations.
    """
    var start_nanos: Int64
    var end_nanos: Int64
    var running: Bool

    fn __init__(inout self):
        """Creates a new stopped timer."""
        self.start_nanos = 0
        self.end_nanos = 0
        self.running = False

    fn start(inout self):
        """Starts or restarts the timer."""
        # Would need: self.start_nanos = system_nanos()
        self.start_nanos = 0  # Placeholder
        self.running = True

    fn stop(inout self):
        """Stops the timer."""
        if self.running:
            # Would need: self.end_nanos = system_nanos()
            self.end_nanos = 0  # Placeholder
            self.running = False

    fn reset(inout self):
        """Resets the timer."""
        self.start_nanos = 0
        self.end_nanos = 0
        self.running = False

    fn elapsed(self) -> Duration:
        """Returns elapsed duration."""
        if self.running:
            # Would need: return Duration(system_nanos() - self.start_nanos)
            return Duration.ZERO
        return Duration(self.end_nanos - self.start_nanos)

    fn is_running(self) -> Bool:
        """Returns True if timer is running."""
        return self.running

# ============================================================================
# Parsing Utilities
# ============================================================================

struct DateTimeParser:
    """Utilities for parsing date/time strings."""

    @staticmethod
    fn parse_date(s: String) -> Date:
        """
        Parses date from string.

        Supported formats:
        - YYYY-MM-DD (ISO 8601)
        - MM/DD/YYYY
        - DD.MM.YYYY
        """
        # Try ISO format first (YYYY-MM-DD)
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            let year = DateTimeParser._parse_int(s, 0, 4)
            let month = DateTimeParser._parse_int(s, 5, 7)
            let day = DateTimeParser._parse_int(s, 8, 10)
            return Date(year, month, day)

        # Try MM/DD/YYYY
        if len(s) >= 10 and s[2] == "/" and s[5] == "/":
            let month = DateTimeParser._parse_int(s, 0, 2)
            let day = DateTimeParser._parse_int(s, 3, 5)
            let year = DateTimeParser._parse_int(s, 6, 10)
            return Date(year, month, day)

        # Try DD.MM.YYYY
        if len(s) >= 10 and s[2] == "." and s[5] == ".":
            let day = DateTimeParser._parse_int(s, 0, 2)
            let month = DateTimeParser._parse_int(s, 3, 5)
            let year = DateTimeParser._parse_int(s, 6, 10)
            return Date(year, month, day)

        # Default to epoch
        return Date(1970, 1, 1)

    @staticmethod
    fn parse_time(s: String) -> Time:
        """
        Parses time from string.

        Supported formats:
        - HH:MM:SS
        - HH:MM
        - HH:MM:SS.sss
        """
        if len(s) >= 5 and s[2] == ":":
            let hour = DateTimeParser._parse_int(s, 0, 2)
            let minute = DateTimeParser._parse_int(s, 3, 5)

            var second = 0
            var nano = 0

            if len(s) >= 8 and s[5] == ":":
                second = DateTimeParser._parse_int(s, 6, 8)

                # Check for fractional seconds
                if len(s) > 9 and s[8] == ".":
                    # Parse up to 9 digits of fractional seconds
                    var frac_end = 9
                    while frac_end < len(s) and s[frac_end] >= "0" and s[frac_end] <= "9":
                        frac_end += 1

                    let frac_str = s[9:frac_end]
                    var frac_val = DateTimeParser._parse_int(frac_str, 0, len(frac_str))

                    # Normalize to nanoseconds
                    let digits = frac_end - 9
                    if digits < 9:
                        for _ in range(9 - digits):
                            frac_val *= 10

                    nano = frac_val

            return Time(hour, minute, second, nano)

        return Time.midnight()

    @staticmethod
    fn parse_datetime(s: String) -> DateTime:
        """
        Parses datetime from string.

        Supported formats:
        - YYYY-MM-DDTHH:MM:SS (ISO 8601)
        - YYYY-MM-DD HH:MM:SS
        """
        # Find separator
        var sep_pos = -1
        for i in range(len(s)):
            if s[i] == "T" or s[i] == " ":
                sep_pos = i
                break

        if sep_pos > 0:
            let date_str = s[0:sep_pos]
            let time_str = s[sep_pos + 1:]

            let date = DateTimeParser.parse_date(date_str)
            let time = DateTimeParser.parse_time(time_str)

            return DateTime(date, time)

        # Only date provided
        let date = DateTimeParser.parse_date(s)
        return DateTime(date, Time.midnight())

    @staticmethod
    fn _parse_int(s: String, start: Int, end: Int) -> Int:
        """Parses integer from substring."""
        var result = 0
        var i = start

        while i < end and i < len(s):
            let c = s[i]
            if c >= "0" and c <= "9":
                result = result * 10 + (ord(c) - ord("0"))
            i += 1

        return result

# ============================================================================
# Utility Functions
# ============================================================================

fn is_leap_year(year: Int) -> Bool:
    """Returns True if year is a leap year."""
    return Date._is_leap_year(year)

fn days_in_month(year: Int, month: Int) -> Int:
    """Returns the number of days in a month."""
    let date = Date(year, month, 1)
    return date.days_in_month()

fn days_in_year(year: Int) -> Int:
    """Returns the number of days in a year."""
    return Date._days_in_year(year)

fn timestamp_now() -> Int64:
    """Returns current Unix timestamp (placeholder - needs system time)."""
    # Would need system time integration
    return 0

fn sleep(duration: Duration):
    """Sleeps for the specified duration (placeholder - needs system call)."""
    # Would need: system_sleep(duration.total_nanos())
    pass

# ============================================================================
# Tests
# ============================================================================

fn test_duration():
    """Test Duration operations."""
    print("Testing Duration...")

    # Construction
    let d1 = Duration.from_secs(90)
    assert_true(d1.total_secs() == 90, "from_secs")
    assert_true(d1.total_mins() == 1, "total_mins")

    let d2 = Duration.from_hms(1, 30, 45)
    assert_true(d2.total_secs() == 5445, "from_hms")

    # Arithmetic
    let d3 = d1 + d2
    assert_true(d3.total_secs() == 5535, "addition")

    let d4 = d2 - d1
    assert_true(d4.total_secs() == 5355, "subtraction")

    # Comparison
    assert_true(d1 < d2, "less than")
    assert_true(d2 > d1, "greater than")

    print("Duration tests passed!")

fn test_date():
    """Test Date operations."""
    print("Testing Date...")

    let d = Date(2026, 1, 15)
    assert_true(d.year == 2026, "year")
    assert_true(d.month == 1, "month")
    assert_true(d.day == 15, "day")

    # Leap year
    assert_true(Date._is_leap_year(2024), "2024 is leap year")
    assert_true(not Date._is_leap_year(2025), "2025 is not leap year")
    assert_true(Date._is_leap_year(2000), "2000 is leap year")
    assert_true(not Date._is_leap_year(1900), "1900 is not leap year")

    # ISO format
    let iso = d.to_iso_string()
    assert_true(iso == "2026-01-15", "ISO format")

    # Day of year
    let jan1 = Date(2026, 1, 1)
    assert_true(jan1.day_of_year() == 1, "Jan 1 day of year")

    let feb1 = Date(2026, 2, 1)
    assert_true(feb1.day_of_year() == 32, "Feb 1 day of year")

    # Date arithmetic
    let tomorrow = d.add_days(1)
    assert_true(tomorrow.day == 16, "add_days")

    let next_month = d.add_months(1)
    assert_true(next_month.month == 2, "add_months")

    print("Date tests passed!")

fn test_time():
    """Test Time operations."""
    print("Testing Time...")

    let t = Time(14, 30, 45)
    assert_true(t.hour == 14, "hour")
    assert_true(t.minute == 30, "minute")
    assert_true(t.second == 45, "second")

    # ISO format
    let iso = t.to_iso_string()
    assert_true(iso == "14:30:45", "ISO format")

    # 12-hour format
    assert_true(t.hour_12() == 2, "12-hour conversion")
    assert_true(t.is_pm(), "is PM")

    let morning = Time(9, 15, 0)
    assert_true(morning.is_am(), "is AM")
    assert_true(morning.hour_12() == 9, "AM 12-hour")

    # Seconds since midnight
    let midnight = Time.midnight()
    assert_true(midnight.to_secs_since_midnight() == 0, "midnight secs")

    let noon = Time.noon()
    assert_true(noon.to_secs_since_midnight() == 43200, "noon secs")

    print("Time tests passed!")

fn test_datetime():
    """Test DateTime operations."""
    print("Testing DateTime...")

    let dt = DateTime(2026, 1, 15, 14, 30, 45)
    assert_true(dt.year() == 2026, "year")
    assert_true(dt.hour() == 14, "hour")

    # ISO format
    let iso = dt.to_iso_string()
    assert_true(iso == "2026-01-15T14:30:45", "ISO format")

    # Timestamp roundtrip
    let ts = dt.to_timestamp()
    let dt2 = DateTime.from_timestamp(ts)
    assert_true(dt2.year() == dt.year(), "timestamp roundtrip year")
    assert_true(dt2.month() == dt.month(), "timestamp roundtrip month")
    assert_true(dt2.day() == dt.day(), "timestamp roundtrip day")

    # Duration arithmetic
    let one_hour = Duration.from_hours(1)
    let dt3 = dt.add(one_hour)
    assert_true(dt3.hour() == 15, "add hour")

    print("DateTime tests passed!")

fn test_parsing():
    """Test date/time parsing."""
    print("Testing parsing...")

    # ISO date
    let d1 = DateTimeParser.parse_date("2026-01-15")
    assert_true(d1.year == 2026, "parse ISO date year")
    assert_true(d1.month == 1, "parse ISO date month")
    assert_true(d1.day == 15, "parse ISO date day")

    # US date format
    let d2 = DateTimeParser.parse_date("01/15/2026")
    assert_true(d2.year == 2026, "parse US date")

    # Time
    let t1 = DateTimeParser.parse_time("14:30:45")
    assert_true(t1.hour == 14, "parse time hour")
    assert_true(t1.minute == 30, "parse time minute")

    # DateTime
    let dt = DateTimeParser.parse_datetime("2026-01-15T14:30:45")
    assert_true(dt.year() == 2026, "parse datetime")
    assert_true(dt.hour() == 14, "parse datetime hour")

    print("Parsing tests passed!")

fn test_timezone():
    """Test Timezone operations."""
    print("Testing Timezone...")

    let utc = Timezone.UTC
    assert_true(utc.offset_minutes == 0, "UTC offset")
    assert_true(utc.offset_string() == "+00:00", "UTC string")

    let est = Timezone.EST
    assert_true(est.offset_minutes == -300, "EST offset")
    assert_true(est.offset_string() == "-05:00", "EST string")

    let jst = Timezone.JST
    assert_true(jst.offset_minutes == 540, "JST offset")
    assert_true(jst.offset_string() == "+09:00", "JST string")

    print("Timezone tests passed!")

fn assert_true(condition: Bool, message: String):
    """Assert helper."""
    if not condition:
        print("ASSERTION FAILED: " + message)

fn run_all_tests():
    """Run all time module tests."""
    print("=== Time Module Tests ===")
    test_duration()
    test_date()
    test_time()
    test_datetime()
    test_parsing()
    test_timezone()
    print("=== All tests passed! ===")
