# Neural Architecture Search using Genetic Algorithm (NAS-GA)

## 📋 Overview

This implementation modifies the base NAS-GA code with two key improvements for Assignment 2:

- **Q1A:** Roulette-wheel selection (replaces tournament selection)
- **Q1B:** Separate penalties for Conv (0.015) and FC (0.008) layers in fitness function

---

## 🚀 Quick Start - Google Colab (RECOMMENDED)

### Why Colab?
- ✅ Free T4 GPU access
- ✅ 1-3 hours runtime (vs 8-15 hours on CPU)
- ✅ No local installation needed

### Step 1: Clone This Repository in Colab

Open Google Colab: https://colab.research.google.com/

Create a new notebook and run:

```python
# Enable GPU first: Runtime → Change runtime type → T4 GPU → Save

# Clone the repository
!git clone https://github.com/GyanStore/AI-Assignments.git
%cd AI-Assignments

# Install dependencies
!pip install torch torchvision --quiet

# Verify GPU
import torch
print("✓ GPU Available:", torch.cuda.is_available())
print("✓ GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

### Step 2: Run the Experiment

```python
# Run NAS-GA (1-3 hours with GPU)
print("="*60)
print("Starting NAS-GA Experiment")
print("Population: 10, Generations: 5")
print("Estimated time: 1-3 hours with GPU")
print("="*60)

!python nas_run.py
```

**⚠️ IMPORTANT:** Keep the Colab tab open during the entire run!

---

## 📁 Output Files - How They're Created & Where to Find Them

### Automatic Output Folder Creation

The code **automatically creates** the output structure:

```
outputs/
└── run_1/                          # Auto-created folder
    ├── nas_run.log                 # Complete execution log (ALL console output)
    ├── generation_0.jsonl          # Architecture genes from generation 0
    ├── generation_1.jsonl          # Architecture genes from generation 1
    ├── generation_2.jsonl          # Architecture genes from generation 2
    ├── generation_3.jsonl          # Architecture genes from generation 3
    ├── generation_4.jsonl          # Architecture genes from generation 4
    └── best_arch.pkl               # Best architecture (pickled)
```

### How It Works (Code Flow):

1. **`nas_run.py` (line 10-13):**
   ```python
   if not os.path.exists('outputs'):
       os.mkdir('outputs')
   # Counts existing runs and creates run_X/
   os.mkdir(f'outputs/run_{len(all_logs)+1}')
   ```

2. **`nas_run.py` (line 17):**
   ```python
   # ALL print statements go to nas_run.log
   sys.stdout = open(f'outputs/run_1/nas_run.log', 'w')
   ```

3. **`model_ga.py` (line 258):**
   ```python
   # Saves architecture genes after each generation
   with open(f'outputs/run_{run}/generation_{generation}.jsonl', 'w') as f:
       ...
   ```

4. **`nas_run.py` (line 55-56):**
   ```python
   # Saves best architecture at the end
   with open(f'outputs/run_1/best_arch.pkl', 'wb') as f:
       pickle.dump(best_arch, f)
   ```

### In Google Colab - Finding Your Logs:

**Method 1: Using File Browser**
1. Click the **📁 folder icon** on the left sidebar
2. Navigate to: `AI-Assignments/` → `outputs/` → `run_1/`
3. You'll see `nas_run.log` appear here
4. Right-click → Download

**Method 2: View Log in Colab**
```python
# View the complete log
!cat outputs/run_1/nas_run.log
```

**Method 3: Download Files Programmatically**
```python
from google.colab import files

# Download log file
files.download('outputs/run_1/nas_run.log')

# Download best architecture
files.download('outputs/run_1/best_arch.pkl')
```

---

## 📊 What's in the Log File?

The `nas_run.log` contains:

```
Using device: cuda
Starting with 10 Population: [Arch(...), Arch(...), ...]

============================================================
Generation 1/5
============================================================
Evaluating architecture 1/10... Fitness: 0.4523, Accuracy: 0.4580
Evaluating architecture 2/10... Fitness: 0.4312, Accuracy: 0.4390
...
Best in generation: Arch(conv=3, acc=0.4580)

Performing roulette-wheel selection of total population: 10 ...
Performing Crossover & Mutation ...
Elitism: Keeping top 2 architectures in next generation.

[continues for all 5 generations...]

============================================================
FINAL BEST ARCHITECTURE
============================================================
Genes: {...}
Accuracy: 0.5234
Fitness: 0.5198

Total parameters: 123,456
Model architecture:
CNN(...)
```

---

## 🔑 Key Modifications (Q1A & Q1B)

### Q1A: Roulette-Wheel Selection

**Location:** `model_ga.py`, lines 135-145

```python
def selection(self):
    """Q1A: Roulette-Wheel Selection based on relative fitness"""
    fitness_sum = sum(arch.fitness for arch in self.population)
    
    if fitness_sum <= 0:
        return random.choices(self.population, k=self.population_size)
    
    probabilities = [arch.fitness / fitness_sum for arch in self.population]
    selected = random.choices(self.population, weights=probabilities, 
                              k=self.population_size)
    return selected
```

**Benefits:**
- Selection probability proportional to fitness
- Better genetic diversity
- All architectures have a chance to be selected

### Q1B: Separate Conv/FC Penalties

**Location:** `model_ga.py`, lines 102-125

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

conv_weight = 0.015  # Higher penalty (1.875x)
fc_weight = 0.008    # Lower penalty

conv_penalty = (conv_params / 1e6) * conv_weight
fc_penalty = (fc_params / 1e6) * fc_weight
complexity_penalty = conv_penalty + fc_penalty

architecture.fitness = best_acc - complexity_penalty
```

**Justification:**
- Conv layers: 2D spatial operations, higher computational cost → Higher penalty (0.015)
- FC layers: Simple matrix multiplication → Lower penalty (0.008)
- Ratio: 1.875× reflects real-world computational complexity difference

---

## ⚙️ Experiment Parameters

```python
population_size = 10        # Number of architectures per generation
generations = 5             # Number of evolutionary generations
mutation_rate = 0.3         # Probability of mutation
crossover_rate = 0.7        # Probability of crossover
training_samples = 5000     # CIFAR-10 training subset
validation_samples = 1000   # CIFAR-10 validation subset
epochs = 100                # Max training epochs per architecture
patience = 10               # Early stopping patience
```

---

## 📸 For Your Report - Screenshots to Take

After execution completes in Colab:

### Screenshot 1: Verify GPU & Start
```python
!head -20 outputs/run_1/nas_run.log
```
Shows: `Using device: cuda`

### Screenshot 2: Roulette-Wheel Selection Evidence
```python
!grep -i "roulette" outputs/run_1/nas_run.log
```
Shows: `Performing roulette-wheel selection...`

### Screenshot 3: Final Results
```python
!tail -30 outputs/run_1/nas_run.log
```
Shows: Final best architecture, accuracy, fitness, parameters

### Screenshot 4: File Structure
```python
!ls -lh outputs/run_1/
```
Shows: All generated files with sizes

---

## 💻 Local Execution (NOT RECOMMENDED - 8-15 hours)

If you must run locally:

```bash
# Install dependencies
pip install torch torchvision

# Run
python nas_run.py
```

**Note:** Will take 8-15 hours on CPU. Use Colab instead!

---

## 🔍 Verifying Modifications

### Check Q1A (Roulette-Wheel Selection):
```bash
grep -A 10 "def selection" model_ga.py
```
Should show: `"""Q1A: Roulette-Wheel Selection based on relative fitness"""`

### Check Q1B (Separate Penalties):
```bash
grep -A 15 "Q1B:" model_ga.py
```
Should show: `conv_weight = 0.015` and `fc_weight = 0.008`

---

## 📦 Files Description

| File | Purpose |
|------|---------|
| `model_ga.py` | Genetic Algorithm with Q1A & Q1B modifications |
| `model_cnn.py` | CNN architecture builder |
| `nas_run.py` | Main execution script |
| `README.md` | This file |

---

## 🎯 Expected Results

- **Runtime:** 1-3 hours with GPU (Colab T4)
- **Best Accuracy:** ~50-60% (5000 training samples, early generations)
- **Output Log:** Complete evolution history saved in `outputs/run_1/nas_run.log`
- **Best Architecture:** Saved in `outputs/run_1/best_arch.pkl`

---

## ❓ Troubleshooting

**Q: I don't see the outputs folder**
- Wait for execution to start. It's created within first second.
- Click refresh icon in Colab's file browser.

**Q: Colab disconnected during run**
- Keep the tab open/active during 1-3 hour run
- Consider using Colab Pro for longer sessions

**Q: Output log is empty**
- The log file only updates after the program completes
- Check Colab cell output for real-time progress

**Q: How to download outputs?**
- Right-click file in Colab → Download
- Or use: `files.download('outputs/run_1/nas_run.log')`

---

## 📚 References

- Original NAS-GA: https://github.com/ayan-cs/nas-ga-basic
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch Documentation: https://pytorch.org/docs/

---

## 📧 Repository

GitHub: https://github.com/GyanStore/AI-Assignments

---

## ✅ Success Checklist

Before submitting your report, verify:

- [ ] Ran with GPU (T4) on Colab
- [ ] Used full parameters (population=10, generations=5)
- [ ] Downloaded `outputs/run_1/nas_run.log`
- [ ] Log shows "roulette-wheel selection"
- [ ] Log shows final best architecture with accuracy & fitness
- [ ] Took screenshots of key outputs
- [ ] Execution completed (all 5 generations)

---

**Good luck with your assignment! 🚀**
