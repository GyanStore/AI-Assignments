# Assignment 2 - Q1 Modifications

## Changes Made to Original NAS-GA Code

### Q1A: Roulette-Wheel Selection
**Modified:** `selection()` function in `model_ga.py` (lines 130-140)

**Original:** Tournament selection with tournament_size=3  
**Modified:** Roulette-wheel selection using proportional fitness probabilities

```python
def selection(self):
    """Q1A: Roulette-Wheel Selection based on relative fitness"""
    fitness_sum = sum(arch.fitness for arch in self.population)
    
    if fitness_sum <= 0:
        return random.choices(self.population, k=self.population_size)
    
    probabilities = [arch.fitness / fitness_sum for arch in self.population]
    selected = random.choices(self.population, weights=probabilities, k=self.population_size)
    
    return selected
```

### Q1B: Modified Fitness Function
**Modified:** `evaluate_fitness()` function in `model_ga.py` (lines 110-134)

**Original:** Single penalty weight (0.01) for all parameters  
**Modified:** Separate weights for Conv (0.015) and FC (0.008) parameters

```python
# Q1B: Calculate model complexity penalty with separate Conv and FC weights
conv_params = 0
fc_params = 0

for name, param in model.named_parameters():
    num_p = param.numel()
    if 'features' in name:
        conv_params += num_p
    elif 'classifier' in name:
        fc_params += num_p

conv_weight = 0.015
fc_weight = 0.008

conv_penalty = (conv_params / 1e6) * conv_weight
fc_penalty = (fc_params / 1e6) * fc_weight
complexity_penalty = conv_penalty + fc_penalty

architecture.fitness = best_acc - complexity_penalty
```

### Weight Justification

**Conv Weight (0.015) - Higher penalty:**
- 2D spatial operations (kernel sliding)
- Higher computational cost per parameter
- Batch normalization overhead

**FC Weight (0.008) - Lower penalty:**
- Simple matrix multiplication
- Lower computational cost per parameter
- More cache-friendly operations

**Ratio:** 1.875× (0.015/0.008) reflects computational complexity difference

## How to Run

```bash
cd nas-ga-basic
python nas_run.py
```

## Expected Output

- Creates `outputs/run_X/` directory
- Generates `nas_run.log` with detailed execution logs
- Shows "roulette-wheel selection" in output
- Saves best architecture in `best_arch.pkl`

## Requirements

```
torch>=1.10.0
torchvision>=0.11.0
```

Install with: `pip install torch torchvision`

