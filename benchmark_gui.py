import customtkinter as ctk
import qai_hub as hub
import threading

from PIL import Image
from tkinter import messagebox
from one_script_to_rule_them_all import inference_dataset, process_results, calculate_accuracy, InputSpec, extract_number, FileType
from os import listdir, path

class BenchmarkQAIHub():
	def __init__(self):
		self.root = ctk.CTk()
		self.root.geometry("400x400")
		self.root.title("Benchmark QAI Hub")
		self.root.iconbitmap("favicon.ico")
		
		self.appraise_image = Image.open("Appraise.png")
		self.title_image = ctk.CTkImage(light_image=self.appraise_image, dark_image=self.appraise_image, size=(308,85))
		
		self.title_label = ctk.CTkLabel(self.root, text="", image=self.title_image)
		self.title_label.pack()

		self.window_sub_label = ctk.CTkLabel(self.root, text="Benchmarking Tool", font=("Arial", 18))
		self.window_sub_label.pack(pady=(0, 20))

		self.model_path_label = ctk.CTkLabel(self.root, text="Model File Path:", font=("Arial", 12))
		self.model_path_label.pack()

		self.model_path_entry = ctk.CTkEntry(self.root, placeholder_text="./model.library", font=("Arial", 16), width=300)
		self.model_path_entry.pack()

		self.device_name_label = ctk.CTkLabel(self.root, text="Device Name:", font=("Arial", 12))
		self.device_name_label.pack()

		self.device_name_entry = ctk.CTkEntry(self.root, placeholder_text="Samsung Galaxy S24 (Family)", font=("Arial", 16), width=300)
		self.device_name_entry.pack()

		self.datasets_dir_label = ctk.CTkLabel(self.root, text="Datasets Directory Path:", font=("Arial", 12))
		self.datasets_dir_label.pack()

		self.datasets_dir_entry = ctk.CTkEntry(self.root, placeholder_text="./datasets_quantized_library", font=("Arial", 16), width=300)
		self.datasets_dir_entry.pack()

		self.button_icon = Image.open("running-icon.png")

		self.run_benchmark_button = ctk.CTkButton(self.root, text="Run Benchmark", command=self.run_benchmark_threaded, image=ctk.CTkImage(dark_image=self.button_icon, light_image=self.button_icon))
		self.run_benchmark_button.pack(pady=25)

		self.root.mainloop()

	def run_benchmark_threaded(self):
		# Run the benchmark in a separate thread to prevent UI freezing.
		messagebox.showinfo(title="WARNING", message="Be careful not to click the button again.\nIt will start up another thread.👍\n- Adolfo")
		threading.Thread(target=self.run_benchmark, daemon=True).start()

	def run_benchmark(self):
		if self.get_dataset_dir() and self.get_model_path() and self.get_device_name():
			try:
				# Upload Model
				model_path = self.model_path_entry.get()
				model = hub.upload_model(model_path)

				# Model ID
				model_id = model.model_id

				# Model Name
				model_name = path.splitext(model.name)[0]

				# Library Name
				model_library = model.model_type.name.lower()

				# Get Device Name
				device_name = self.device_name_entry.get()

				# Set benchmark results directory
				model_device = device_name.replace(" ", "").lower()
				results_dir = f"{model_name}_{model_library}_{model_device}"

				### Run inference on image datasets
				# Dataset Paths
				datasets_dir = self.datasets_dir_entry.get()
				dataset_paths = [
					f"./{datasets_dir}/" + image_dataset 
					for image_dataset in listdir(f"./{datasets_dir}")
				]
				dataset_paths.sort(key=extract_number)

				# Split into two lists with alternating elements
				list1 = dataset_paths[::2]  # odd
				list2 = dataset_paths[1::2]  # even

				thread1 = threading.Thread(
				    target=inference_dataset,
				    args=(list1, model_id, device_name, model_name, results_dir)
				)

				thread2 = threading.Thread(
				    target=inference_dataset,
				    args=(list2, model_id, device_name, model_name, results_dir)
				)

				thread1.start()
				thread2.start()

				thread1.join()
				thread2.join()

				### Process results from inference
				result_paths = [f"./{results_dir}/" + result for result in listdir(f"./{results_dir}")]
				result_paths.sort(key=extract_number)
				process_results(result_paths, "./class_index.json", "./synset.json")

				### Calculate accuracy based on processed results
				results_json = "results.json"
				ground_truth_json = "ground_truth.json"

				calculate_accuracy(results_json, ground_truth_json, device_name, model_name, model_library)

				# Show success message after completion
				messagebox.showinfo(title="✅ Success!", message="Benchmark completed successfully!")

			except Exception as e:
				messagebox.showerror(title="❌ Error!", message=f"An error occurred: {e}")
				messagebox.showerror(title="❌ Error!", message="Bro. Btw. Check the textbox inputs if that other error message made zero sense. 👍")

	def get_model_path(self):
		model_path = self.model_path_entry.get()
		if not model_path:
			messagebox.showerror(title="❌ Error!", message="No model file path given.")
			return False
		return True

	def get_device_name(self):
		device_name = self.device_name_entry.get()
		if not device_name:
			messagebox.showerror(title="❌ Error!", message="No device name given.")
			return False
		return True

	def get_dataset_dir(self):
		datasets_dir = self.datasets_dir_entry.get()
		if not datasets_dir:
			messagebox.showerror(title="❌ Error!", message="No datasets directory path given.")
			return False
		return True

BenchmarkQAIHub()