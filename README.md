## Create Conda environment first

```bash
conda create python=3.10 -n benchmark_qai_hub
```

## Activate environment

```bash
conda activate benchmark_qai_hub
```

## Install requirements
```bash
pip install -r  .\requirements.txt
```

### Run GUI
```bash
python. .\benchmark_gui.py
```

### Notes
Preferably your paths should look like this

```
📂 BenchmarkQAIHub/
├─── 📁 datasets_onnx
├─── 📁 datasets_quantized_onnx
├─── 📁 datasets_quantized_tflite
├─── 📁 datasets_tflite
├─── 🐍 benchmark_gui.py
├─── 🐍 one_script_to_rule_them_all.py
├─── 📃 class_index.json
├─── 📃 ground_truth.json
├─── 📃 synset.json
├─── 📄 requirements.txt
├─── 📄 model_accuracy_scores.txt
└─── etc.
```

## To Do
- [ ] Keep an individual log of inference fails
- [ ] Add button to recalculate accuracy given results directory
- [X] Change `Datasets Directory Path` entry to accept full paths
- [ ] 