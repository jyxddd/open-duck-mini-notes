"""Apply the duck changes to a clean xiaozhi-esp32 v2.4.0 tree (run from its root)."""
import pathlib
import shutil

HERE = pathlib.Path(__file__).parent


def sub(path, pairs):
    p = pathlib.Path(path)
    s = p.read_text(encoding="utf8")
    for a, b in pairs:
        assert s.count(a) == 1, (path, a)
        s = s.replace(a, b)
    p.write_text(s, encoding="utf8")


for f in ("duck_local.h", "duck_local.cc"):
    shutil.copy(HERE / f, "main/" + f)
sub("main/CMakeLists.txt", [('"mcp_server.cc"', '"mcp_server.cc"\n            "duck_local.cc"')])
sub("CMakeLists.txt", [('set(PROJECT_VER "2.4.0")', 'set(PROJECT_VER "9.4.0")  # above any release, OTA leaves it alone')])
sub("main/audio/wake_words/custom_wake_word.cc", [
    ('#include <esp_mn_speech_commands.h>\n', '#include <esp_mn_speech_commands.h>\n#include "duck_local.h"\n'),
    ('        commands_.push_back({CONFIG_CUSTOM_WAKE_WORD, CONFIG_CUSTOM_WAKE_WORD_DISPLAY, "wake"});\n',
     '        commands_.push_back({CONFIG_CUSTOM_WAKE_WORD, CONFIG_CUSTOM_WAKE_WORD_DISPLAY, "wake"});\n'
     '        commands_.push_back({"ni hao xiao zhi", "你好小智", "wake"});\n'
     '        commands_.push_back({"ting xia", "停下", "duck_stop"});\n'),
    ('                if (command.action == "wake") {\n',
     '                if (command.action == "duck_stop") {\n'
     '                    duck_local::StopDuck();\n'
     '                } else if (command.action == "wake") {\n'),
])
# keep listening for "ting xia" while it talks and listens; stock does this only for
# AFE wake words, so check on the board that the multinet really hears anything then
sub("main/application.cc", [
    ('                // Only AFE wake word can be detected in speaking mode\n'
     '                audio_service_.EnableWakeWordDetection(audio_service_.IsAfeWakeWord());\n',
     '                audio_service_.EnableWakeWordDetection(true);\n'),
    ('    // Enable wake word detection in listening mode (configured via Kconfig)\n'
     '    audio_service_.EnableWakeWordDetection(audio_service_.IsAfeWakeWord());\n',
     '    // Enable wake word detection in listening mode (configured via Kconfig)\n'
     '    audio_service_.EnableWakeWordDetection(true);\n'),
])
sub("main/boards/waveshare/esp32-s3-audio-board/esp32-s3-audio_board.cc", [
    ('#include "config.h"\n', '#include "config.h"\n#include "duck_local.h"\n'),
    ('        InitializeButtons();\n', '        InitializeButtons();\n        duck_local::AddTools();\n'),
])
sub("main/boards/waveshare/esp32-s3-audio-board/config.json", [
    ('"CONFIG_USE_WECHAT_MESSAGE_STYLE=y",\n',
     '"CONFIG_USE_WECHAT_MESSAGE_STYLE=y",\n'
     '                "CONFIG_USE_CUSTOM_WAKE_WORD=y",\n'
     '                "CONFIG_CUSTOM_WAKE_WORD=\\"ni hao xiao dou\\"",\n'
     '                "CONFIG_CUSTOM_WAKE_WORD_DISPLAY=\\"你好小豆\\"",\n'
     '                "CONFIG_CUSTOM_WAKE_WORD_THRESHOLD=20",\n'
     '                "CONFIG_WAKE_WORD_DETECTION_IN_LISTENING=y",\n'
     '                "CONFIG_SR_MN_CN_MULTINET7_QUANT=y",\n'),
])
print("applied")
