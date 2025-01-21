# oda-r (On Device AI Reasoning)

oda-r is a professional-grade compiler for Declarative Self-improving Python (DSPy), featuring comprehensive error handling, logging, and configuration management.


## Features

- Async support for improved performance
- Comprehensive error handling and logging
- Configuration file support
- Enhanced prompt management
- Sophisticated verification strategies
- Metrics collection
- Progress feedback

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mitkox/oda-r.git
cd oda-r
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python odar.py <path_to_dspy_file> [--config CONFIG] [--debug]
```

Example:
```bash
python odar.py reasoning_questions.dspy --question 1 --debug
```

## Configuration

The compiler can be configured using a YAML configuration file. You can specify:

- Server URL
- Model parameters (temperature, top_p, etc.)
- Maximum tokens
- Timeouts and iteration limits

Example configuration can be found in `config.yaml`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
