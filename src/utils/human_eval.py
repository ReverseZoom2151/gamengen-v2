"""
Human Evaluation Framework for GameNGen
Based on paper Section 5.1: Human Evaluation

"We provided 10 human raters with 130 random short clips (of lengths 1.6 seconds
and 3.2 seconds) of our simulation side by side with the real game."
"""

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional


@dataclass
class EvaluationClip:
    """Single evaluation clip"""

    clip_id: int
    duration_seconds: float
    real_video_path: str
    fake_video_path: str
    real_is_on_left: bool  # Randomize which side is real


@dataclass
class EvaluationResult:
    """Result from single evaluation"""

    clip_id: int
    duration_seconds: float
    user_choice: str  # "left" or "right"
    correct: bool
    confidence: int  # 1-5 scale
    time_taken_seconds: float


@dataclass
class BlindedResponse:
    """A rater response that contains no correctness information."""

    clip_id: int
    user_choice: str
    confidence: int
    time_taken_seconds: float


class HumanEvaluationFramework:
    """
    Framework for conducting human evaluation studies

    Paper methodology:
    - 10 human raters
    - 130 clips (1.6s and 3.2s lengths)
    - Side-by-side comparison
    - Task: Identify which is the real game
    """

    def __init__(
        self,
        output_dir: str = "human_eval_results",
        clip_lengths: Optional[List[float]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clip_lengths = clip_lengths or [1.6, 3.2]
        self.evaluation_clips: List[EvaluationClip] = []

    def create_evaluation_clips(
        self,
        real_videos: List[str],
        fake_videos: List[str],
        num_clips_per_length: int = 65,  # 130 total / 2 lengths
        seed: Optional[int] = None,
    ) -> List[EvaluationClip]:
        """
        Create evaluation clip pairs

        Args:
            real_videos: Paths to real gameplay videos
            fake_videos: Paths to generated gameplay videos
            num_clips_per_length: Number of clips per duration

        Returns:
            List of evaluation clips
        """
        if not real_videos or not fake_videos:
            raise ValueError("real_videos and fake_videos must both be non-empty")
        rng = random.Random(seed)
        clips = []
        clip_id = 0

        for duration in self.clip_lengths:
            for _ in range(num_clips_per_length):
                # Randomly select videos
                real_video = rng.choice(real_videos)
                fake_video = rng.choice(fake_videos)

                # Randomize which side is real
                real_on_left = rng.choice([True, False])

                clip = EvaluationClip(
                    clip_id=clip_id,
                    duration_seconds=duration,
                    real_video_path=real_video,
                    fake_video_path=fake_video,
                    real_is_on_left=real_on_left,
                )

                clips.append(clip)
                clip_id += 1

        self.evaluation_clips = clips
        return clips

    def save_evaluation_protocol(self, filename: str = "evaluation_protocol.json") -> tuple[Path, Path]:
        """Save a blinded public protocol and a separate answer key."""
        protocol = {
            "format_version": 2,
            "num_clips": len(self.evaluation_clips),
            "clip_lengths": self.clip_lengths,
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "duration_seconds": clip.duration_seconds,
                    "left_video_path": self._public_media_path(clip.clip_id, "left", clip.real_video_path if clip.real_is_on_left else clip.fake_video_path),
                    "right_video_path": self._public_media_path(clip.clip_id, "right", clip.fake_video_path if clip.real_is_on_left else clip.real_video_path),
                }
                for clip in self.evaluation_clips
            ],
        }

        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(protocol, f, indent=2)
        answer_key = self.output_dir / "answer_key.json"
        with open(answer_key, "w", encoding="utf-8") as f:
            json.dump({str(clip.clip_id): clip.real_is_on_left for clip in self.evaluation_clips}, f, indent=2)

        print(f"Saved blinded evaluation protocol to {output_path}")
        return output_path, answer_key

    def _public_media_path(self, clip_id: int, side: str, source_path: str) -> str:
        suffix = Path(source_path).suffix.lower() or ".mp4"
        return f"media/clip_{clip_id:04d}_{side}{suffix}"

    def materialize_blinded_media(self) -> Path:
        """Copy source clips to neutral public names before serving the UI."""
        media_dir = self.output_dir / "media"
        media_dir.mkdir(exist_ok=True)
        for clip in self.evaluation_clips:
            for side, source in (
                ("left", clip.real_video_path if clip.real_is_on_left else clip.fake_video_path),
                ("right", clip.fake_video_path if clip.real_is_on_left else clip.real_video_path),
            ):
                source_path = Path(source)
                if not source_path.is_file():
                    raise FileNotFoundError(f"cannot materialize missing evaluation clip: {source_path}")
                destination = self.output_dir / self._public_media_path(clip.clip_id, side, source)
                shutil.copy2(source_path, destination)
        return media_dir

    def build_web_ui(self, protocol_filename: str = "evaluation_protocol.json") -> Path:
        """Create a static browser UI that exports unscored responses."""
        path = self.output_dir / "evaluate.html"
        protocol_json = json.dumps(protocol_filename)
        html = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>GameNGen blinded evaluation</title>
<style>body{font:16px system-ui;margin:2rem}.pair{display:flex;gap:1rem}video{width:48%;background:#111}section{margin:2rem 0}button{margin:.25rem}</style>
<h1>Which clip is real?</h1><p>Select a side and confidence, then export unscored responses.</p>
<label>Rater ID <input id="rater" required></label><div id="clips"></div><button id="export">Export responses</button>
<script>
const started=performance.now(),responses=new Map();
fetch(PROTOCOL).then(r=>r.json()).then(protocol=>{
 document.querySelector('#clips').innerHTML=protocol.clips.map(c=>`<section data-id="${c.clip_id}"><h2>Clip ${c.clip_id+1} (${c.duration_seconds}s)</h2><div class="pair"><video controls src="${c.left_video_path}"></video><video controls src="${c.right_video_path}"></video></div><button data-choice="left">Left is real</button><button data-choice="right">Right is real</button><label> Confidence <select><option value="">Choose</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label></section>`).join('');
 document.querySelectorAll('section button').forEach(button=>button.onclick=()=>{const section=button.closest('section'),confidence=Number(section.querySelector('select').value);if(!confidence)return alert('Select confidence first.');responses.set(Number(section.dataset.id),{clip_id:Number(section.dataset.id),user_choice:button.dataset.choice,confidence,time_taken_seconds:(performance.now()-started)/1000});section.style.outline='3px solid #3a3';});
});
document.querySelector('#export').onclick=()=>{const id=document.querySelector('#rater').value.trim();if(!id)return alert('Enter rater ID.');const body={format_version:1,evaluator_id:id,responses:[...responses.values()]},blob=new Blob([JSON.stringify(body,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`responses_${id}.json`;a.click();};
</script></html>""".replace("PROTOCOL", protocol_json)
        path.write_text(html, encoding="utf-8")
        return path

    def score_blinded_responses(
        self, response_payload: Mapping[str, Any], answer_key_path: str | Path | None = None
    ) -> List[EvaluationResult]:
        """Apply the private answer key only after responses are collected."""
        key_path = Path(answer_key_path or self.output_dir / "answer_key.json")
        answer_key = json.loads(key_path.read_text(encoding="utf-8"))
        clips = {clip.clip_id: clip for clip in self.evaluation_clips}
        seen = set()
        results = []
        for raw in response_payload.get("responses", []):
            response = BlindedResponse(**raw)
            if response.clip_id in seen:
                raise ValueError(f"duplicate response for clip {response.clip_id}")
            if response.clip_id not in clips or str(response.clip_id) not in answer_key:
                raise ValueError(f"response references unknown clip {response.clip_id}")
            if response.user_choice not in {"left", "right"}:
                raise ValueError("user_choice must be left or right")
            if not 1 <= response.confidence <= 5 or response.time_taken_seconds < 0:
                raise ValueError("response confidence or duration is invalid")
            seen.add(response.clip_id)
            real_is_left = bool(answer_key[str(response.clip_id)])
            results.append(EvaluationResult(
                clip_id=response.clip_id,
                duration_seconds=clips[response.clip_id].duration_seconds,
                user_choice=response.user_choice,
                correct=(response.user_choice == "left") == real_is_left,
                confidence=response.confidence,
                time_taken_seconds=response.time_taken_seconds,
            ))
        return results

    def run_evaluation_session(
        self, evaluator_id: str, start_clip_idx: int = 0, input_fn: Callable[[str], str] = input
    ) -> List[EvaluationResult]:
        """
        Run evaluation session with human rater

        Args:
            evaluator_id: Unique identifier for this evaluator
            start_clip_idx: Clip to start from (for resuming)

        Returns:
            List of evaluation results
        """
        print("=" * 70)
        print("GameNGen Human Evaluation Session")
        print("=" * 70)
        print(f"Evaluator ID: {evaluator_id}")
        print(f"Total clips: {len(self.evaluation_clips)}")
        print(f"Starting from clip: {start_clip_idx}")
        print("\nInstructions:")
        print("- You will see two videos side by side")
        print("- One is the real game, one is neural simulation")
        print("- Press 'L' if you think LEFT is real")
        print("- Press 'R' if you think RIGHT is real")
        print("- Press 'Q' to quit and save progress")
        print("=" * 70 + "\n")

        results = []

        for i in range(start_clip_idx, len(self.evaluation_clips)):
            clip = self.evaluation_clips[i]

            print(
                f"\nClip {i+1}/{len(self.evaluation_clips)} "
                f"(Duration: {clip.duration_seconds}s)"
            )

            start_time = time.time()

            print(f"  Present left/right clips for clip {clip.clip_id} using the UI layer.")
            choice = input_fn("  Which side is real? [L/R, Q to quit]: ").strip().lower()

            if choice == "q":
                print("\nQuitting and saving progress...")
                break

            elapsed = time.time() - start_time

            # Record result
            if choice in ["l", "r"]:
                correct = (choice == "l" and clip.real_is_on_left) or (
                    choice == "r" and not clip.real_is_on_left
                )

                confidence_text = input_fn("  Confidence [1-5]: ").strip()
                if confidence_text not in {"1", "2", "3", "4", "5"}:
                    print("  Invalid confidence; response not recorded.")
                    continue
                confidence = int(confidence_text)

                result = EvaluationResult(
                    clip_id=clip.clip_id,
                    duration_seconds=clip.duration_seconds,
                    user_choice=choice,
                    correct=correct,
                    confidence=confidence,
                    time_taken_seconds=elapsed,
                )

                results.append(result)
            else:
                print("  Invalid choice; response not recorded.")

        # Save results
        self.save_results(evaluator_id, results)

        # Print summary
        self.print_summary(results)

        return results

    def save_results(self, evaluator_id: str, results: List[EvaluationResult]):
        """Save evaluation results"""
        results_data = {
            "evaluator_id": evaluator_id,
            "timestamp": time.time(),
            "num_clips_evaluated": len(results),
            "results": [asdict(r) for r in results],
        }

        filename = self.output_dir / f"results_{evaluator_id}.json"
        with open(filename, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults saved to {filename}")

    def print_summary(self, results: List[EvaluationResult]):
        """Print evaluation summary"""
        if not results:
            return

        correct_count = sum(1 for r in results if r.correct)
        accuracy = correct_count / len(results) * 100

        # By duration
        durations = set(r.duration_seconds for r in results)
        duration_stats = {}

        for dur in durations:
            dur_results = [r for r in results if r.duration_seconds == dur]
            dur_correct = sum(1 for r in dur_results if r.correct)
            dur_accuracy = dur_correct / len(dur_results) * 100 if dur_results else 0
            duration_stats[dur] = {
                "count": len(dur_results),
                "correct": dur_correct,
                "accuracy": dur_accuracy,
            }

        print("\n" + "=" * 70)
        print("Evaluation Summary:")
        print("=" * 70)
        print(f"Total clips evaluated: {len(results)}")
        print(f"Correct identifications: {correct_count}/{len(results)}")
        print(f"Overall accuracy: {accuracy:.1f}%")
        print(f"Paper reference: 58% (1.6s), 60% (3.2s)")
        print()

        for dur, stats in sorted(duration_stats.items()):
            print(f"{dur}s clips:")
            print(
                f"  Accuracy: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['count']})"
            )

        print("=" * 70)

    def analyze_results(self, results_dir: Optional[str] = None) -> dict:
        """
        Analyze all evaluation results

        Args:
            results_dir: Directory with result JSON files

        Returns:
            Aggregated statistics
        """
        if results_dir is None:
            results_dir = self.output_dir

        results_dir = Path(results_dir)

        # Load all result files
        all_results = []
        evaluator_count = 0

        for result_file in results_dir.glob("results_*.json"):
            with open(result_file, "r") as f:
                data = json.load(f)
                all_results.extend(data["results"])
                evaluator_count += 1

        if not all_results:
            print("No results found")
            return {}

        # Convert to EvaluationResult objects
        results = [EvaluationResult(**r) for r in all_results]

        # Compute statistics
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        accuracy = correct / total * 100 if total > 0 else 0

        # By duration
        duration_stats = {}
        for dur in self.clip_lengths:
            dur_results = [r for r in results if abs(r.duration_seconds - dur) < 0.1]
            if dur_results:
                dur_correct = sum(1 for r in dur_results if r.correct)
                duration_stats[dur] = {
                    "total": len(dur_results),
                    "correct": dur_correct,
                    "accuracy": dur_correct / len(dur_results) * 100,
                }

        stats = {
            "evaluators": evaluator_count,
            "total_clips": total,
            "correct": correct,
            "accuracy": accuracy,
            "by_duration": duration_stats,
        }

        # Print report
        print("\n" + "=" * 70)
        print("Human Evaluation Analysis")
        print("=" * 70)
        print(f"Evaluators: {evaluator_count}")
        print(f"Total evaluations: {total}")
        print(f"Overall accuracy: {accuracy:.1f}%")
        print(f"\nPaper results: 58% (1.6s), 60% (3.2s)")
        print()

        for dur, data in duration_stats.items():
            print(
                f"{dur}s clips: {data['accuracy']:.1f}% ({data['correct']}/{data['total']})"
            )

        print("=" * 70)

        return stats


if __name__ == "__main__":
    # Demo usage
    print("Human Evaluation Framework Demo")

    framework = HumanEvaluationFramework()

    # Create dummy clips
    print("\nCreating evaluation protocol...")
    clips = framework.create_evaluation_clips(
        real_videos=["real1.mp4", "real2.mp4"],
        fake_videos=["fake1.mp4", "fake2.mp4"],
        num_clips_per_length=5,
    )

    print(f"Created {len(clips)} evaluation clips")

    # Save protocol
    framework.save_evaluation_protocol()

    print("\nFramework ready!")
    print("To run actual evaluation, use:")
    print("  framework.run_evaluation_session('evaluator_001')")
