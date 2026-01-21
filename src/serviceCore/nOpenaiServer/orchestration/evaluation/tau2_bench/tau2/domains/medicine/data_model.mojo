# Medicine Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the medicine domain

from collections import Dict, List

# Appointment status constants
alias APPOINTMENT_STATUS_SCHEDULED = "scheduled"
alias APPOINTMENT_STATUS_COMPLETED = "completed"
alias APPOINTMENT_STATUS_CANCELLED = "cancelled"

# Prescription status constants
alias PRESCRIPTION_STATUS_ACTIVE = "active"
alias PRESCRIPTION_STATUS_EXPIRED = "expired"
alias PRESCRIPTION_STATUS_FILLED = "filled"


struct Patient:
    """Represents a patient."""
    var patient_id: String
    var first_name: String
    var last_name: String
    var dob: String
    var email: String
    var phone: String
    var insurance_id: String
    var medical_history: List[String]
    
    fn __init__(out self, patient_id: String, first_name: String, last_name: String,
                dob: String, email: String):
        self.patient_id = patient_id
        self.first_name = first_name
        self.last_name = last_name
        self.dob = dob
        self.email = email
        self.phone = ""
        self.insurance_id = ""
        self.medical_history = List[String]()
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct Doctor:
    """Represents a doctor."""
    var doctor_id: String
    var first_name: String
    var last_name: String
    var specialty: String
    var department: String
    var available_slots: List[String]
    
    fn __init__(out self, doctor_id: String, first_name: String, last_name: String,
                specialty: String):
        self.doctor_id = doctor_id
        self.first_name = first_name
        self.last_name = last_name
        self.specialty = specialty
        self.department = ""
        self.available_slots = List[String]()
    
    fn full_name(self) -> String:
        return "Dr. " + self.first_name + " " + self.last_name


struct Appointment:
    """Represents a medical appointment."""
    var appointment_id: String
    var patient_id: String
    var doctor_id: String
    var date: String
    var time: String
    var status: String
    var notes: String
    
    fn __init__(out self, appointment_id: String, patient_id: String, 
                doctor_id: String, date: String, time: String):
        self.appointment_id = appointment_id
        self.patient_id = patient_id
        self.doctor_id = doctor_id
        self.date = date
        self.time = time
        self.status = APPOINTMENT_STATUS_SCHEDULED
        self.notes = ""


struct Prescription:
    """Represents a prescription."""
    var prescription_id: String
    var patient_id: String
    var doctor_id: String
    var medication: String
    var dosage: String
    var frequency: String
    var start_date: String
    var end_date: String
    var status: String
    
    fn __init__(out self, prescription_id: String, patient_id: String,
                doctor_id: String, medication: String, dosage: String):
        self.prescription_id = prescription_id
        self.patient_id = patient_id
        self.doctor_id = doctor_id
        self.medication = medication
        self.dosage = dosage
        self.frequency = ""
        self.start_date = ""
        self.end_date = ""
        self.status = PRESCRIPTION_STATUS_ACTIVE


struct MedicineDB:
    """Database for the medicine domain."""
    var patients: Dict[String, Patient]
    var doctors: Dict[String, Doctor]
    var appointments: Dict[String, Appointment]
    var prescriptions: Dict[String, Prescription]
    
    fn __init__(out self):
        self.patients = Dict[String, Patient]()
        self.doctors = Dict[String, Doctor]()
        self.appointments = Dict[String, Appointment]()
        self.prescriptions = Dict[String, Prescription]()
    
    fn add_patient(inout self, patient: Patient):
        self.patients[patient.patient_id] = patient
    
    fn add_doctor(inout self, doctor: Doctor):
        self.doctors[doctor.doctor_id] = doctor
    
    fn add_appointment(inout self, appointment: Appointment):
        self.appointments[appointment.appointment_id] = appointment
    
    fn get_patient(self, patient_id: String) raises -> Patient:
        if patient_id not in self.patients:
            raise Error("Patient " + patient_id + " not found")
        return self.patients[patient_id]
    
    fn get_doctor(self, doctor_id: String) raises -> Doctor:
        if doctor_id not in self.doctors:
            raise Error("Doctor " + doctor_id + " not found")
        return self.doctors[doctor_id]

