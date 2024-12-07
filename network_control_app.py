import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from threading import Thread, Event
from network_simulation.network import NodeNetwork
from network_simulation.gui_visualizer import GUIVisualizer
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

matplotlib.use("TkAgg")


class NetworkControlApp:
    def __init__(self, root, num_nodes, initial_connections, alpha=1.7):
        self.root = root
        self.root.title("Network Control Panel")

        # Simulation parameters
        self.num_nodes = num_nodes
        self.initial_connections = initial_connections
        self.alpha = alpha

        # Shared variables
        self.epsilon = tk.DoubleVar(value=0.4)
        self.display_interval = tk.IntVar(value=1)
        self.running = Event()  # Simulation running state (paused initially)

        # Input fields to track changes
        self.epsilon_input = tk.StringVar(value=str(self.epsilon.get()))
        self.display_interval_input = tk.StringVar(value=str(self.display_interval.get()))
        self.changes_pending = False

        # Network and Visualizer
        self.network = NodeNetwork(
            num_nodes=self.num_nodes,
            num_connections=self.initial_connections,
            alpha=self.alpha,
            epsilon=self.epsilon.get()
        )
        self.visualizer = GUIVisualizer(self.network)

        # Create UI
        self.create_widgets()

        # Start simulation in a separate thread
        self.simulation_thread = Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()

    def create_widgets(self):
        # Main layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Frame for controls and display
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="NSEW")
        main_frame.columnconfigure(1, weight=1)

        # Control Panel
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=0, column=0, sticky="NS")
        control_frame.columnconfigure(0, weight=1)

        # Epsilon Control
        ttk.Label(control_frame, text="Epsilon:").grid(row=0, column=0, sticky="W")
        epsilon_entry = ttk.Entry(control_frame, textvariable=self.epsilon_input, width=10)
        epsilon_entry.grid(row=0, column=1, sticky="EW")
        epsilon_entry.bind("<KeyRelease>", self.on_input_change)

        # Display Interval Control
        ttk.Label(control_frame, text="Display Interval:").grid(row=1, column=0, sticky="W")
        interval_entry = ttk.Entry(control_frame, textvariable=self.display_interval_input, width=10)
        interval_entry.grid(row=1, column=1, sticky="EW")
        interval_entry.bind("<KeyRelease>", self.on_input_change)

        # Add/Remove Nodes Buttons
        ttk.Label(control_frame, text="Modify Nodes:").grid(row=2, column=0, sticky="W")
        node_buttons_frame = ttk.Frame(control_frame)
        node_buttons_frame.grid(row=2, column=1, sticky="EW")
        add_nodes_button = ttk.Button(node_buttons_frame, text="Add Nodes", command=self.add_nodes)
        add_nodes_button.grid(row=0, column=0, padx=5)
        remove_nodes_button = ttk.Button(node_buttons_frame, text="Remove Nodes", command=self.remove_nodes)
        remove_nodes_button.grid(row=0, column=1, padx=5)

        # Apply/Cancel Buttons
        action_buttons_frame = ttk.Frame(control_frame)
        action_buttons_frame.grid(row=3, column=0, columnspan=2, pady=10)
        apply_button = ttk.Button(action_buttons_frame, text="Apply Changes", command=self.apply_changes)
        apply_button.grid(row=0, column=0, padx=5)
        cancel_button = ttk.Button(action_buttons_frame, text="Cancel Changes", command=self.cancel_changes)
        cancel_button.grid(row=0, column=1, padx=5)

        # Play and Pause Buttons
        ttk.Label(control_frame, text="Simulation Control:").grid(row=4, column=0, sticky="W")
        control_buttons_frame = ttk.Frame(control_frame)
        control_buttons_frame.grid(row=4, column=1, sticky="EW")
        play_button = ttk.Button(control_buttons_frame, text="Play", command=self.start_simulation)
        play_button.grid(row=0, column=0, padx=5)
        pause_button = ttk.Button(control_buttons_frame, text="Pause", command=self.pause_simulation)
        pause_button.grid(row=0, column=1, padx=5)

        # Quit Button
        quit_button = ttk.Button(control_frame, text="Quit", command=self.quit_application)
        quit_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Network Visualization
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=1, sticky="NSEW")
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # Matplotlib Canvas for Visualization
        self.canvas = FigureCanvasTkAgg(self.visualizer.visualizer.fig, master=display_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="NSEW")
        self.canvas.draw()

    def run_simulation(self):
        step = 0
        while True:
            if self.running.is_set():
                # Update network state
                self.network.update_network()

                # Display network at intervals
                if step % self.display_interval.get() == 0:
                    self.network.apply_forces(min(50, self.display_interval.get()))
                    self.visualizer.show(step)
                    self.canvas.draw()

                time.sleep(0.001)  # Small delay to avoid busy looping
                step += 1
            else:
                time.sleep(0.1)  # Sleep briefly when paused

    def on_input_change(self, event):
        """Highlight text fields when their values differ from current settings."""
        self.changes_pending = False
        if float(self.epsilon_input.get()) != self.epsilon.get():
            event.widget.config(background="lightyellow")
            self.changes_pending = True
        else:
            event.widget.config(background="white")

        if int(self.display_interval_input.get()) != self.display_interval.get():
            event.widget.config(background="lightyellow")
            self.changes_pending = True
        else:
            event.widget.config(background="white")

    def apply_changes(self):
        """Apply changes to epsilon and display interval."""
        self.epsilon.set(float(self.epsilon_input.get()))
        self.display_interval.set(int(self.display_interval_input.get()))
        print(f"Applied changes: Epsilon={self.epsilon.get()}, Display Interval={self.display_interval.get()}")

    def cancel_changes(self):
        """Reset inputs to current settings."""
        self.epsilon_input.set(str(self.epsilon.get()))
        self.display_interval_input.set(str(self.display_interval.get()))
        print("Canceled changes.")

    def add_nodes(self):
        """Add nodes to the network."""
        add_count = simpledialog.askinteger("Add Nodes", "How many nodes to add?", minvalue=1, parent=self.root)
        if add_count:
            self.network.add_nodes(add_count)
            self.num_nodes += add_count
            self.visualizer.update_network(self.network)
            print(f"Added {add_count} nodes. Total nodes: {self.num_nodes}")

    def remove_nodes(self):
        """Remove nodes from the network."""
        remove_count = simpledialog.askinteger("Remove Nodes", "How many nodes to remove?", minvalue=1, maxvalue=self.num_nodes - 1, parent=self.root)
        if remove_count:
            self.network.remove_nodes(remove_count)
            self.num_nodes -= remove_count
            self.visualizer.update_network(self.network)
            print(f"Removed {remove_count} nodes. Total nodes: {self.num_nodes}")

    def start_simulation(self):
        self.running.set()

    def pause_simulation(self):
        self.running.clear()

    def quit_application(self):
        self.running.clear()
        self.root.destroy()


# Main Application
if __name__ == "__main__":
    num_nodes = 200
    initial_connections = 1200
    alpha = 1.7

    root = tk.Tk()
    app = NetworkControlApp(root, num_nodes, initial_connections, alpha=alpha)
    root.mainloop()