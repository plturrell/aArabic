use rand::Rng;
use crate::data::math_generator::MathProblem;

pub struct MoiraiGenerator {
    rng: rand::rngs::ThreadRng,
}

impl MoiraiGenerator {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    pub fn generate(&mut self, difficulty: &str) -> MathProblem {
        let p_type = self.rng.gen_range(0..3);
        match p_type {
            0 => self.generate_patch_projection(difficulty),
            1 => self.generate_attention(difficulty),
            _ => self.generate_distribution(difficulty),
        }
    }

    fn generate_patch_projection(&mut self, _difficulty: &str) -> MathProblem {
        let seq_len = self.rng.gen_range(100..5000);
        let patch_size = match self.rng.gen_range(0..3) {
            0 => 8, 1 => 16, _ => 32
        };
        
        let num_patches = seq_len / patch_size;
        let remainder = seq_len % patch_size;

        MathProblem {
            question: format!("A Moirai model processes a time series of length L={}. If the patch size P={}, how many complete patches N are generated, and how many tokens remain?", seq_len, patch_size),
            answer: format!("{}, {}", num_patches, remainder),
            solution: format!("N = floor(L/P) = floor({}/{}) = {}. Remainder = {} % {} = {}.", seq_len, patch_size, num_patches, seq_len, patch_size, remainder),
            category: "moirai".to_string(),
            operation: "patch_projection".to_string(),
            difficulty: _difficulty.to_string(),
        }
    }

    fn generate_attention(&mut self, _difficulty: &str) -> MathProblem {
        let num_variates = self.rng.gen_range(2..100);
        let time_steps = self.rng.gen_range(10..200);
        let total_tokens = num_variates * time_steps;

        MathProblem {
            question: format!("Calculte the total number of tokens for Any-Variate Attention given {} variates each having {} time steps.", num_variates, time_steps),
            answer: total_tokens.to_string(),
            solution: format!("Total tokens = Variates * TimeSteps = {} * {} = {}.", num_variates, time_steps, total_tokens),
            category: "moirai".to_string(),
            operation: "any_variate_attention".to_string(),
            difficulty: _difficulty.to_string(),
        }
    }

    fn generate_distribution(&mut self, _difficulty: &str) -> MathProblem {
        // Student's t-distribution degrees of freedom
        let n = self.rng.gen_range(10..100); // sample size
        let k = 1; // parameters
        let df = n - k;

        MathProblem {
            question: format!("For a Student's t-distribution mixture head estimated from {} samples with {} parameter (mean), what are the degrees of freedom (nu)?", n, k),
            answer: df.to_string(),
            solution: format!("Degrees of freedom nu = n - k = {} - {} = {}.", n, k, df),
            category: "moirai".to_string(),
            operation: "mixture_distribution".to_string(),
            difficulty: _difficulty.to_string(),
        }
    }
}
