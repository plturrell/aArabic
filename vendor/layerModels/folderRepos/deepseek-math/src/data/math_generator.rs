use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MathProblem {
    pub question: String,
    pub answer: String,
    pub solution: String,
    pub category: String,
    pub operation: String,
    pub difficulty: String,
}

pub struct MathOperationGenerator {
    rng: rand::rngs::ThreadRng,
}

impl MathOperationGenerator {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    pub fn generate_problem(&mut self, category: &str, difficulty: &str) -> Option<MathProblem> {
        match category {
            "arithmetic" => Some(self.generate_arithmetic(difficulty)),
            "algebra" => Some(self.generate_algebra(difficulty)),
            "geometry" => Some(self.generate_geometry(difficulty)),
            "number_theory" => Some(self.generate_number_theory(difficulty)),
            _ => None,
        }
    }

    fn generate_arithmetic(&mut self, difficulty: &str) -> MathProblem {
        let (a, b) = match difficulty {
            "hard" => (self.rng.gen_range(100..1000), self.rng.gen_range(100..1000)),
            "medium" => (self.rng.gen_range(10..100), self.rng.gen_range(10..100)),
            _ => (self.rng.gen_range(1..10), self.rng.gen_range(1..10)),
        };

        // Randomly choose operation + - *
        let op = self.rng.gen_range(0..3); 
        let (op_symbol, res) = match op {
            0 => ("+", a + b),
            1 => ("-", a - b),
            _ => ("*", a * b),
        };

        MathProblem {
            question: format!("Calculate {} {} {}.", a, op_symbol, b),
            answer: res.to_string(),
            solution: format!("{} {} {} = {}", a, op_symbol, b, res),
            category: "arithmetic".to_string(),
            operation: "basic_arithmetic".to_string(),
            difficulty: difficulty.to_string(),
        }
    }

    fn generate_algebra(&mut self, difficulty: &str) -> MathProblem {
        // Solve for x: ax + b = c
        let x = self.rng.gen_range(1..20);
        let a = self.rng.gen_range(2..10);
        let b = self.rng.gen_range(1..50);
        let c = a * x + b;

        MathProblem {
            question: format!("Solve for x: {}x + {} = {}.", a, b, c),
            answer: x.to_string(),
            solution: format!("{}x + {} = {} -> {}x = {} -> x = {}", a, b, c, a, c - b, x),
            category: "algebra".to_string(),
            operation: "linear_equation".to_string(),
            difficulty: difficulty.to_string(),
        }
    }

    fn generate_geometry(&mut self, _difficulty: &str) -> MathProblem {
        // Area of rectangle
        let w = self.rng.gen_range(1..50);
        let h = self.rng.gen_range(1..50);
        let area = w * h;

        MathProblem {
            question: format!("Find the area of a rectangle with width {} and height {}.", w, h),
            answer: area.to_string(),
            solution: format!("Area = width * height = {} * {} = {}.", w, h, area),
            category: "geometry".to_string(),
            operation: "area_rectangle".to_string(),
            difficulty: _difficulty.to_string(),
        }
    }

    fn generate_number_theory(&mut self, _difficulty: &str) -> MathProblem {
        // GCD
        let val = self.rng.gen_range(1..50);
        let a = val * self.rng.gen_range(1..10);
        let b = val * self.rng.gen_range(1..10);
        
        // Simple GCD implementation
        fn gcd(mut a: i32, mut b: i32) -> i32 {
            while b != 0 {
                let temp = b;
                b = a % b;
                a = temp;
            }
            a
        }
        
        let res = gcd(a, b);

        MathProblem {
            question: format!("Find the Greatest Common Divisor (GCD) of {} and {}.", a, b),
            answer: res.to_string(),
            solution: format!("GCD({}, {}) = {}", a, b, res),
            category: "number_theory".to_string(),
            operation: "gcd".to_string(),
            difficulty: _difficulty.to_string(),
        }
    }
}
