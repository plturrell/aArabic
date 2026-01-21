# School Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the school domain

from collections import Dict, List

# Enrollment status constants
alias ENROLLMENT_STATUS_ENROLLED = "enrolled"
alias ENROLLMENT_STATUS_DROPPED = "dropped"
alias ENROLLMENT_STATUS_WAITLISTED = "waitlisted"
alias ENROLLMENT_STATUS_COMPLETED = "completed"


struct Course:
    """Represents a course."""
    var course_id: String
    var name: String
    var description: String
    var department: String
    var credits: Int
    var instructor_id: String
    var capacity: Int
    var enrolled: Int
    var schedule: String
    
    fn __init__(out self, course_id: String, name: String, department: String,
                credits: Int):
        self.course_id = course_id
        self.name = name
        self.description = ""
        self.department = department
        self.credits = credits
        self.instructor_id = ""
        self.capacity = 30
        self.enrolled = 0
        self.schedule = ""
    
    fn available_seats(self) -> Int:
        return self.capacity - self.enrolled
    
    fn is_full(self) -> Bool:
        return self.enrolled >= self.capacity


struct Instructor:
    """Represents an instructor."""
    var instructor_id: String
    var first_name: String
    var last_name: String
    var email: String
    var department: String
    var courses: List[String]
    
    fn __init__(out self, instructor_id: String, first_name: String, 
                last_name: String, email: String):
        self.instructor_id = instructor_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.department = ""
        self.courses = List[String]()
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct Enrollment:
    """Represents a course enrollment."""
    var enrollment_id: String
    var student_id: String
    var course_id: String
    var status: String
    var grade: String
    var enrolled_at: String
    
    fn __init__(out self, enrollment_id: String, student_id: String, 
                course_id: String):
        self.enrollment_id = enrollment_id
        self.student_id = student_id
        self.course_id = course_id
        self.status = ENROLLMENT_STATUS_ENROLLED
        self.grade = ""
        self.enrolled_at = ""


struct Student:
    """Represents a student."""
    var student_id: String
    var first_name: String
    var last_name: String
    var email: String
    var major: String
    var year: Int
    var gpa: Float64
    var enrollments: List[String]
    
    fn __init__(out self, student_id: String, first_name: String, 
                last_name: String, email: String):
        self.student_id = student_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.major = ""
        self.year = 1
        self.gpa = 0.0
        self.enrollments = List[String]()
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct SchoolDB:
    """Database for the school domain."""
    var courses: Dict[String, Course]
    var instructors: Dict[String, Instructor]
    var students: Dict[String, Student]
    var enrollments: Dict[String, Enrollment]
    
    fn __init__(out self):
        self.courses = Dict[String, Course]()
        self.instructors = Dict[String, Instructor]()
        self.students = Dict[String, Student]()
        self.enrollments = Dict[String, Enrollment]()
    
    fn add_course(inout self, course: Course):
        self.courses[course.course_id] = course
    
    fn add_student(inout self, student: Student):
        self.students[student.student_id] = student
    
    fn add_instructor(inout self, instructor: Instructor):
        self.instructors[instructor.instructor_id] = instructor
    
    fn get_course(self, course_id: String) raises -> Course:
        if course_id not in self.courses:
            raise Error("Course " + course_id + " not found")
        return self.courses[course_id]
    
    fn get_student(self, student_id: String) raises -> Student:
        if student_id not in self.students:
            raise Error("Student " + student_id + " not found")
        return self.students[student_id]

