"""Column names shared by postprocessing without importing the GPU runtime."""
BASIC = ['dice', 'iou', 'precision', 'recall', 'cldice', 'hd95_mm']
SKELETON = ['skeleton_recall', 'skeleton_precision', 'cldice_recomputed',
            'cldice_absolute_difference', 'branch_detection_rate_80',
            'branch_detection_rate_90', 'branch_coverage_mean', 'gt_branch_count',
            'detected_branch_count_80', 'detected_branch_count_90',
            'excluded_short_branch_count', 'excluded_short_branch_total_length_mm',
            'excluded_cycle_component_count', 'degenerate_structure_count']
AV = ['av_mcs', 'artery_to_vein_rate', 'vein_to_artery_rate',
      'merged_vessel_dice', 'merged_vessel_precision', 'merged_vessel_recall']
