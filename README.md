## Constraint-Based-Quest-Generation Environment Setup

This README will walk you through downloading, installing, configuring the melonDS Nintendo DS emulator on your machine, and how to customize the controls to play our game.

---

## 💾 Downloading melonDS

1. Visit the official melonDS website:
   👉 [https://melonds.kuribo64.net/](https://melonds.kuribo64.net/)

2. Navigate to the **Downloads** section.

3. Choose the appropriate version for your operating system:
   - **Windows:** Download the `.zip` or `.7z` file.
   - **Linux/macOS:** Follow platform-specific instructions or build from source if necessary.

4. Extract the contents of the downloaded file to a folder of your choice.

---

## 🛠️ Initial Setup

1. **Run melonDS**:
   - On Windows, run `melonDS.exe`.
   - On Linux/macOS, execute the binary through terminal or supported GUI.

2. **BIOS/firmware files**:
   - melonDS requires BIOS and firmware dumps from a real DS to function fully. These files include:
     - `bios7.bin`
     - `bios9.bin`
     - `firmware.bin`
   - Place them in the same directory as the melonDS executable or configure their paths in the emulator settings.


## 🎮 Configuring Controls

You can customize the control bindings to match your preferred input device (keyboard or gamepad):

1. Launch melonDS.
2. Click **Config > Input and Hotkeys**.
3. A new window will appear showing the default controls.
4. Click on a control (e.g., **A**, **B**, **Up**, **Down**) and press the key or button you want to assign.
5. Repeat this process for all controls, including touch screen and hotkeys.
6. Click **OK** to save your settings.

We recommend mapping **W**, **S**, **A**, **D** to **X**, **B**, **Y**, **A**, and the directional pad to your arrow keys. 

---

## 🎮 Playing our HGSS ROM

This repository includes a sample `CQGv1.1.0.nds` ROM file for testing and gameplay.

### Steps:

1. Download the ROM file:
   - Go to the [Releases](./releases) section or browse the repository files.
   - Download the file named `CQGv1.1.0.nds` (or whatever the most recent version is).

2. Open the game in melonDS:
   - Launch the emulator.
   - Go to **File > Open ROM**.
   - Select the downloaded `.nds` file.
   - The game should start running immediately.

> ⚠️ **Legal Notice**: The included ROM is provided with permission and is intended only for demonstration or educational use. Do not redistribute or use ROMs you do not have the legal rights to.




Happy gaming! 🎉
