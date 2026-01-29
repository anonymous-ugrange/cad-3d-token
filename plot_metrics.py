import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def load_and_process_json_files(test_dir):
    file_epoch_pairs = []

    for epoch_dir in os.listdir(test_dir):
        try:
            epoch = int(epoch_dir.split("_")[-1])
            json_path = os.path.join(
                test_dir, epoch_dir, "indicator", "mean_report_expert.json"
            )
            file_epoch_pairs.append((epoch, json_path))
        except Exception as e:
            print(e)
    
    file_epoch_pairs.sort(key=lambda x: x[0])
    
    epochs = []
    metrics_data = []
    
    for epoch, file_path in file_epoch_pairs:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                metrics_data.append(data)
                epochs.append(epoch)
                print(f"loading: {os.path.basename(file_path)} (epoch {epoch})")
        except Exception as e:
            print(e)
    
    return epochs, metrics_data

def extract_metrics(epochs, metrics_data):
    line_f1 = [data['line']['f1'] for data in metrics_data]
    arc_f1 = [data['arc']['f1'] for data in metrics_data]
    circle_f1 = [data['circle']['f1'] for data in metrics_data]
    extrusion_f1 = [data['extrusion']['f1'] for data in metrics_data]
    
    cd_median = [data['cd']['median'] for data in metrics_data]
    cd_mean = [data['cd']['mean'] for data in metrics_data]
    
    invalidity_ratio = [data['invalidity_ratio_percentage'] for data in metrics_data]

    metrics_dict = {
        'epochs': epochs,
        'f1_scores': {
            'line': line_f1,
            'arc': arc_f1,
            'circle': circle_f1,
            'extrusion': extrusion_f1
        },
        'cd_scores': {
            'median': cd_median,
            'mean': cd_mean
        },
        'invalidity_ratio': invalidity_ratio
    }
    
    return metrics_dict

def plot_f1_curves(metrics_dict, save_path='f1_curves.png'):
    epochs = metrics_dict['epochs']
    f1_scores = metrics_dict['f1_scores']
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, f1_scores['line'], marker='o', linewidth=2, markersize=4, 
             label=f'Line F1 (final: {f1_scores["line"][-1]:.2f})')
    plt.plot(epochs, f1_scores['arc'], marker='s', linewidth=2, markersize=4, 
             label=f'Arc F1 (final: {f1_scores["arc"][-1]:.2f})')
    plt.plot(epochs, f1_scores['circle'], marker='^', linewidth=2, markersize=4, 
             label=f'Circle F1 (final: {f1_scores["circle"][-1]:.2f})')
    plt.plot(epochs, f1_scores['extrusion'], marker='d', linewidth=2, markersize=4, 
             label=f'Extrusion F1 (final: {f1_scores["extrusion"][-1]:.2f})')
    
    plt.title('F1 Scores Evolution', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('F1 Score (%)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"F1 figure saved to: {save_path}")
    return save_path

def plot_cd_curves(metrics_dict, save_path='cd_curves.png'):
    epochs = metrics_dict['epochs']
    cd_scores = metrics_dict['cd_scores']
    
    plt.figure(figsize=(12, 8))
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('CD Median', fontsize=14, color=color)
    line1 = ax1.plot(epochs, cd_scores['median'], marker='o', color=color, 
                     linewidth=2, markersize=4, label=f'CD Median (final: {cd_scores["median"][-1]:.4f})')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('CD Mean', fontsize=14, color=color)
    line2 = ax2.plot(epochs, cd_scores['mean'], marker='s', color=color, 
                     linewidth=2, markersize=4, label=f'CD Mean (final: {cd_scores["mean"][-1]:.2f})')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=12)
    
    plt.title('Chamfer Distance Evolution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"CD figure saved to: {save_path}")
    return save_path

def plot_invalidity_ratio(metrics_dict, save_path='invalidity_ratio_curve.png'):
    epochs = metrics_dict['epochs']
    invalidity_ratio = metrics_dict['invalidity_ratio']
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, invalidity_ratio, marker='^', linewidth=2, markersize=6, 
             color='purple', label=f'Invalidity Ratio (final: {invalidity_ratio[-1]:.2f}%)')
    
    plt.title('Invalidity Ratio Evolution', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Invalidity Ratio (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"IR figure saved to: {save_path}")
    return save_path

def plot_all_metrics_combined(metrics_dict, save_path='all_metrics_combined.png'):
    epochs = metrics_dict['epochs']
    f1_scores = metrics_dict['f1_scores']
    cd_scores = metrics_dict['cd_scores']
    invalidity_ratio = metrics_dict['invalidity_ratio']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Metrics Overview', fontsize=18, fontweight='bold')
    
    ax1.plot(epochs, f1_scores['line'], marker='o', linewidth=2, label='Line F1')
    ax1.plot(epochs, f1_scores['arc'], marker='s', linewidth=2, label='Arc F1')
    ax1.plot(epochs, f1_scores['circle'], marker='^', linewidth=2, label='Circle F1')
    ax1.set_title('F1 Scores (Line, Arc, Circle)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('F1 Score (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, f1_scores['extrusion'], marker='d', linewidth=2, color='green', label='Extrusion F1')
    ax2.set_title('Extrusion F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(epochs, cd_scores['median'], marker='o', linewidth=2, label='CD Median')
    ax3.plot(epochs, cd_scores['mean'], marker='s', linewidth=2, label='CD Mean')
    ax3.set_title('Chamfer Distance')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Chamfer Distance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(epochs, invalidity_ratio, marker='^', linewidth=2, color='purple', label='Invalidity Ratio')
    ax4.set_title('Invalidity Ratio')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Invalidity Ratio (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"composite metrics saved to: {save_path}")
    return save_path

def main():
    test_dir = "draw2cad_log/my_exp.3/test"
    output_dir = "draw2cad_log/my_exp.3/test/metric_figure"

    result = load_and_process_json_files(test_dir)
    epochs, metrics_data = result
    
    if not epochs:
        print("no valid jsons!")
        return
    
    print(f"successfully process {len(epochs)} epochs")
    print(f"Epoch range: {min(epochs)} - {max(epochs)}")
    
    metrics_dict = extract_metrics(epochs, metrics_data)
    
    os.makedirs(output_dir, exist_ok=True)

    f1_plot = plot_f1_curves(metrics_dict, os.path.join(output_dir, "f1_curves.png"))
    cd_plot = plot_cd_curves(metrics_dict, os.path.join(output_dir, "cd_curves.png"))
    invalidity_plot = plot_invalidity_ratio(metrics_dict, os.path.join(output_dir, "invalidity_ratio.png"))
    combined_plot = plot_all_metrics_combined(metrics_dict, os.path.join(output_dir, "all_metrics_combined.png"))

if __name__ == "__main__":
    main()