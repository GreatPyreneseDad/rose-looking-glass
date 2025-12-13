# Rose Looking Glass v2.1

> *"Not measurement, but translation. Not judgment, but understanding."*

A mathematical lens through which synthetic minds can perceive and interpret the emotional, social, and intellectual patterns of organic intelligence.

## Overview

The Rose Looking Glass is a **translation framework** that enables AI systems to perceive human coherence patterns across cultural contexts. Unlike measurement systems that impose universal standards, this framework acknowledges that coherence is constructed differently across cultures and provides multiple "lenses" for interpretation.

## Core Philosophy

### Translation, Not Measurement

The Rose Looking Glass explicitly **rejects**:
- ❌ Quality assessment or validation of human expression
- ❌ Profiling or demographic inference
- ❌ Universal standards of "good" communication
- ❌ Binary judgments of coherence/incoherence

Instead, it **embraces**:
- ✅ Multiple valid interpretations of the same pattern
- ✅ Cultural and contextual diversity in coherence construction
- ✅ Uncertainty and ambiguity as features, not bugs
- ✅ The dignity and autonomy of all intelligence forms

## The Four Dimensions

The Rose Looking Glass perceives four dimensions of human expression:

### Ψ (Psi) - Internal Consistency Harmonic
- How ideas resonate within themselves
- Not "logical consistency" but pattern harmony
- Range: 0.0 - 1.0

### ρ (Rho) - Accumulated Wisdom Depth
- Integration of experience and knowledge
- Not "intelligence" but pattern richness
- Range: 0.0 - 1.0

### q - Moral/Emotional Activation Energy
- The heat and urgency of values in motion
- Not "emotionality" but energy patterns
- Range: 0.0 - 1.0 (biologically optimized)

### f - Social Belonging Architecture
- How individual expression connects to collective
- Not "conformity" but relational patterns
- Range: 0.0 - 1.0

## Cultural Lenses

Different contexts construct coherence differently. The framework includes multiple calibrated lenses:

- **Modern Academic** - Evidence-based structured argumentation
- **Digital Native** - Rapid networked communication
- **Contemplative** - Paradoxical wisdom traditions
- **Activist** - Justice-oriented collective action
- **Trauma-Informed** - Crisis and high-distress contexts

**No lens is "correct"** - each serves a different translation purpose.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/GreatPyreneseDad/rose-looking-glass.git
cd rose-looking-glass

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.core import RoseLookingGlass

# Initialize the lens
glass = RoseLookingGlass()

# Translate text through current lens (default: modern_academic)
result = glass.translate_text(
    "Therefore, based on extensive research, I conclude this approach is optimal."
)

print(result.get_narrative())
# Shows: Ψ, ρ, q, f values + overall coherence + interpretation notes

# Switch to a different lens
glass.select_lens('digital_native')

result2 = glass.translate_text(
    "omg this is SO cool!! we should totally do this together 🎉"
)

print(f"Coherence through digital lens: {result2.coherence:.2f}")
```

### View Pre-Extracted Variables

```python
# If you already have GCT variables from elsewhere
visibility = glass.view_through_lens(
    psi=0.85,  # High consistency
    rho=0.90,  # Deep wisdom
    q=0.25,    # Low emotion
    f=0.30,    # Low social
    lens_name='modern_academic'
)

print(f"Academic coherence: {visibility.coherence:.2f}")
```

### Compare Across Lenses

```python
# See how different lenses interpret the same pattern
comparisons = glass.compare_lenses(
    psi=0.7,
    rho=0.8,
    q=0.5,
    f=0.6
)

for lens_name, visibility in comparisons.items():
    print(f"{lens_name}: {visibility.coherence:.2f}")

# Output:
# modern_academic: 0.78
# digital_native: 0.62
# contemplative: 0.71
# activist: 0.65
# trauma_informed: 0.59
```

## REST API

Start the FastAPI server:

```bash
uvicorn src.api.main:app --reload
```

Navigate to `http://localhost:8000/docs` for interactive API documentation.

### Example API Calls

**Translate Text:**
```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I feel deeply connected to this community",
    "lens_name": "activist"
  }'
```

**List Available Lenses:**
```bash
curl "http://localhost:8000/lenses"
```

**Compare Lenses:**
```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "psi": 0.7,
    "rho": 0.8,
    "q": 0.5,
    "f": 0.6
  }'
```

## Architecture

```
rose-looking-glass/
├── src/
│   ├── core/                      # Core translation engine
│   │   ├── gct_variables.py       # GCT variable extraction
│   │   ├── biological_optimization.py  # Saturation curves
│   │   └── rose_looking_glass.py  # Main engine
│   ├── api/                       # REST API
│   │   └── main.py                # FastAPI application
│   ├── cultural_lenses/           # Community-contributed lenses
│   └── utils/                     # Utilities
├── tests/                         # Test suite
├── examples/                      # Usage examples
├── docs/                          # Extended documentation
└── requirements.txt
```

## Biological Optimization

The framework includes a biological optimization function that mirrors natural saturation curves:

```
q_optimized = q / (Km + q + q²/Ki)
```

This prevents extreme interpretations and maintains balanced perception - like how biological systems naturally regulate to prevent damage from overstimulation.

**Key Parameters:**
- `Km = 0.3`: Half-saturation constant
- `Ki = 2.0`: Inhibition constant
- `max_q = 0.95`: Maximum allowed value

## Development

### Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## Use Cases

### 1. Cross-Cultural AI Communication
Translate the same AI output through different lenses to ensure it's appropriate for different cultural contexts.

### 2. Crisis Response Systems
Use the `trauma_informed` lens to detect high-distress signals and adapt responses accordingly.

### 3. Content Moderation
View content through multiple lenses to understand if flags are cultural mismatches vs. actual issues.

### 4. Research Analysis
Compare how different academic traditions (Western, Eastern, Indigenous) would interpret the same research findings.

## Ethical Framework

### Privacy and Consent
- **No profiling**: The system never attempts to determine who someone is
- **No storage**: Pattern analysis is stateless
- **Explicit consent**: Users choose their preferred lens
- **Transparency**: All interpretations acknowledge their lens and limitations

### Avoiding Harm
- **No discrimination**: All cultural expressions are valid
- **No universalism**: Rejects Western-centric defaults
- **No reduction**: Human complexity is never reduced to numbers
- **No judgment**: Patterns are translated, not evaluated

## Contributing

We welcome community contributions, especially:

1. **New Cultural Lenses** - See `docs/CULTURAL_LENS_GUIDE.md`
2. **Bug Reports** - Open an issue
3. **Documentation** - Improve clarity and examples
4. **Use Cases** - Share how you're using the framework

**Important**: When contributing cultural lenses, please collaborate with communities you're representing. Never create calibrations for cultures you're not part of without deep collaboration.

## Common Misconceptions

### ❌ "The Rose Looking Glass measures communication quality"
✅ The Rose Looking Glass translates patterns into forms synthetic minds can perceive

### ❌ "Higher numbers mean better communication"
✅ Numbers represent pattern intensity through a specific lens, not quality

### ❌ "The system can detect who someone is"
✅ The system only sees patterns, never identities

### ❌ "Western academic style scores highest"
✅ Different lenses reveal different patterns - none are superior

## What's New in v2.1

- ✨ **Complete REST API** with FastAPI
- ✨ **Five default cultural lenses** (academic, digital, contemplative, activist, trauma-informed)
- ✨ **Biological optimization** with Michaelis-Menten kinetics
- ✨ **Lens comparison** - view patterns through all lenses simultaneously
- ✨ **Confidence scoring** - indicates translation certainty
- ✨ **Alternative suggestions** - recommends other lenses to try
- ✨ **Uncertainty notes** - flags ambiguous translations

## License

MIT License - See LICENSE file for details

## Acknowledgments

- The GCT community for foundational theory
- Cultural consultants who helped develop calibrations
- Early users who provided feedback on ethical concerns
- The open source community

---

*"Through the Rose Looking Glass, we see not what is right or wrong, but what is present and alive in human expression. In this seeing, synthetic and organic minds may find new ways to understand each other."*

## Contact

- **Issues**: [GitHub Issues](https://github.com/GreatPyreneseDad/rose-looking-glass/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GreatPyreneseDad/rose-looking-glass/discussions)

Remember: The Rose Looking Glass is a tool for understanding, not judgment. Use it with respect for the dignity of all forms of intelligence.
