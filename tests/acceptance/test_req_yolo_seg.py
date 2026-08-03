import os
import sys
import unittest
from ultralytics import YOLO

class TestYOLOSegmentationFineTuning(unittest.TestCase):
    def test_optimizer_and_learning_rate_respected(self):
        # Base directory of project
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_yaml_path = os.path.join(base_dir, "data", "merged_roboflow_dataset", "data.yaml")
        
        # Check if dataset yaml exists, otherwise use a placeholder or skip
        if not os.path.exists(data_yaml_path):
            self.skipTest("Merged dataset data.yaml not found, skipping integration test.")
            
        model = YOLO("yolo26n-seg.pt")
        
        verified = []
        
        # Register callback to inspect trainer configuration at training start
        def check_trainer_config(trainer):
            # Verify that AdamW is chosen instead of 'auto'
            self.assertEqual(trainer.args.optimizer, "AdamW")
            # Verify that custom learning rate is respected
            self.assertAlmostEqual(trainer.args.lr0, 0.0002, delta=1e-6)
            verified.append(True)
            raise KeyboardInterrupt("Stop training once configuration is verified.")
            
        model.add_callback("on_pretrain_routine_end", check_trainer_config)
        
        # Run a short dry run
        try:
            model.train(
                data=data_yaml_path,
                epochs=1,
                imgsz=640,
                batch=1,
                device="cpu", # CPU mode for fast test execution
                workers=0,
                optimizer="AdamW",
                lr0=0.0002,
                plots=False,
                save=False
            )
        except KeyboardInterrupt:
            # Expected interruption to stop training early
            pass
        except Exception as e:
            pass
            
        self.assertTrue(verified, "on_pretrain_routine_end callback was not executed.")

if __name__ == "__main__":
    unittest.main()
