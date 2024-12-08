import tkinter as tk
from tkinter import ttk

class MetricsDisplay(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Metrics", padding="10")
        self.text_widget = tk.Text(self, height=10, width=35, wrap="word", state="disabled", bg="lightgray")
        self.text_widget.grid(row=0, column=0, sticky="NSEW")

    def update_metrics(self, metrics_text):
        """Update the metrics displayed in the text widget."""
        self.text_widget.config(state="normal")
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, metrics_text)
        self.text_widget.config(state="disabled")
