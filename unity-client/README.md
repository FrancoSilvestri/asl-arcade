# Unity client

The game half of ASL Arcade: a 3D arcade room where each cabinet teaches part of the
American Sign Language alphabet.

**The Unity source is not in this repository.** The client was written by Daniel
Estrada, and his project is not mine to publish. What is published is the Windows
build the team submitted at the event, attached to the
[v1.0.0 release](https://github.com/FrancoSilvestri/asl-arcade/releases/tag/v1.0.0)
as `asl-arcade-windows-build.zip`.

This page documents how it behaves and the contract it speaks, so the service in
[`../vision-service/`](../vision-service/) can be understood and maintained without
opening the game.

## How it plays

1. Title screen, "ASL LEARNER", with a Play button.
2. An in-world dialog asks for the address of the inference service, prefilled with
   `http://127.0.0.1:8000`. Enter yours and press Apply.
3. You walk around an arcade room in first or third person. Several cabinets are
   scattered around it.
4. Walk into a cabinet and it prompts "Press P to play".
5. The cabinet shows a target letter and an ASL alphabet chart, with your webcam
   feed beside it. Make the sign.
6. Frames stream to the service. When it confirms the sign, the cabinet answers
   "Correct" and awards points. Press Q to step away.

If no camera is available the game reports "No webcam detected!".

## The contract it speaks

The client posts one frame at a time to the **root path** of the address you gave it:

```
POST /
{"frame": "<base64 JPEG>", "target": "A"}
```

and deserialises exactly six fields:

| Field | Meaning |
|---|---|
| `message` | ignored by the client |
| `width`, `height`, `channels` | frame dimensions echoed back |
| `conf` | confidence of the match, `0` when there is none |
| `target_detected` | whether the player made the requested sign |

The current service serves this on `POST /` as a compatibility endpoint and keeps
`POST /detect` as the canonical one. See the
[service README](../vision-service/README.md) for both.

## Running it

1. Download and unzip `asl-arcade-windows-build.zip` from the release.
2. Start the inference service on port 8000, following
   [`../vision-service/README.md`](../vision-service/README.md).
3. Connect a webcam.
4. Run `ASL.exe`, and confirm the address in the Enter IP dialog.

Windows x86-64 only. The build ships the Unity player, so nothing else is needed.

## What it was built with

Unity, with `Newtonsoft.Json` for the HTTP payloads, Cinemachine for the cameras,
the Unity Input System, TextMeshPro for the UI, and the StarterAssets first and
third person controllers.

Built at ShellHacks 2024 in 36 hours, alongside the detector and the service.
