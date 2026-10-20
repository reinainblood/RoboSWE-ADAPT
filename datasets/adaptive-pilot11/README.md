# RoboSWE-ADAPT prospective pilot 11

This runnable task manifest selects eleven native RoboCasa365 tasks for five-trial adaptive evaluation. Each task specifies a robot, scene seed, split and environment arguments. The agent keeps its code and conversation across trials; the simulator resets between trials. The manifest plans 55 trials before environment admission.

The full [83-entry selection covering 85 native tasks](../adaptive-release/selection.json) remains the broader curation record. This pilot provides a smaller starting set for community runs without replacing that catalogue.

## Tasks

| Task | Activity |
|---|---|
| BeverageSorting | Sort drinks between cabinets |
| DateNight | Set the table |
| HotDogSetup | Gather and plate hot dog ingredients |
| ReturnWashingSupplies | Return cleaning supplies |
| PrepareCocktailStation | Gather drink preparation supplies |
| BlendIngredients | Place fruit in the blender and activate it |
| ToastBagel | Prepare a bagel in the toaster oven |
| GetToastedBread | Retrieve toasted bread and serve it |
| LoadDishwasher | Load dishes into the dishwasher |
| ToastOneSlotPair | Move bread to the working toaster slot pair and toast both slices |
| JuiceFruitReamer | Press sliced fruit against a reamer |

MultistepSteaming, PackIdenticalLunches and RecycleSodaCans remain outside this initial pilot because independent reset fingerprints differed during environment checks. StirVegetables is held for verifier review. All four remain in the full curation record. These decisions concern environment repeatability and grading, not model performance.

## Build and admit your scenes

Use the [provisioning workflow](../../docs/provisioning.md) to obtain the pinned official sources and all six asset groups: textures, generated textures, fixtures, Objaverse objects, generated objects and Lightwheel objects. The source references are recorded in `dataset.json`:

- RoboCasa `456174f62b89b8fca99eaaf33949c29fec9cfc2a`
- robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`

Build simulator and code images in your own Docker installation or Modal account. Keep the returned image receipt with the run. No maintainer-owned image, private endpoint, or workstation path is required by this dataset. Asset download links and upstream terms come from the pinned official source; the image and admission receipts identify the actual installation used.

Use the following command sequence. Admission checks the simulator without calling a model and writes the scene manifest used for the paid run.

```sh
roboswe prepare --source-root .
roboswe prepare --source-root . --run --out provisioning.json
roboswe build-images --source-root . -e docker --out images.json
# Configure the participant and budgets using the public run configuration guide.
roboswe admit datasets/adaptive-pilot11 --config run.json --images images.json \
  -e docker --out admitted.json
roboswe run admitted.json --config run.json --images images.json -e docker
```

For Modal, build in your selected profile/environment and use `-e modal` for admission and execution. The [CLI guide](../../docs/adaptive-cli.md) describes provider settings, budgets and execution configuration. Admission records two independent resets and five diagnostic steps per reset, with exact initial fingerprint matching and remote artifact readback for Modal. Rejected and unattempted tasks remain in `admitted.json.rejected.json`; the output also preserves the source and admitted task/trial counts. Report that coverage alongside results. Admission outcomes are not model failures.

## Scene revisions and comparisons

This manifest creates **prospective community scenes**. It deliberately contains no inherited initial fingerprints. Identical task names and seeds do not guarantee identical scenes when asset inventories, runtime versions or platforms differ. Adding generated objects can change the objects sampled by a seeded reset.

The earlier matched pilot used four recorded profiles: a base-asset Modal image, that image plus generated textures, a further image with generated objects, and a separate Docker base-asset image. Those profiles retain their own immutable image identities and fingerprints in their run evidence. This prospective dataset does not substitute a new image into those runs or claim to reproduce their scenes.

Use the admitted scene manifest and its image/source/asset provenance when comparing models. Models compared on the same scene revision should use the same initial fingerprints and trial settings. Publish results from a newly built or differently admitted scene set as a distinct scene revision. Environment admission establishes a starting condition; functional scores still come from the task predicate and independent replay described in the [harness and verification guide](../../docs/harness-and-verification.md).
