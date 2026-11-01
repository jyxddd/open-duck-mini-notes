#include "duck_local.h"

#include <esp_log.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <algorithm>
#include <string>

#include "application.h"
#include "board.h"
#include "mcp_server.h"

#define TAG "DuckLocal"

namespace duck_local {
namespace {

// Same address the phone page uses.
constexpr const char* PI_CMD_URL = "http://<树莓派IP>:8765/cmd";
constexpr int MAX_TIMER_S = 6 * 3600;
// The question takes longer to say, so the model cannot chain start and confirm.
constexpr int64_t CONFIRM_MIN_US = 2 * 1000000LL;
constexpr int64_t CONFIRM_MAX_US = 60 * 1000000LL;
constexpr int IDLE_RETRIES = 5;
constexpr int64_t IDLE_RETRY_US = 1000000LL;

esp_timer_handle_t timer = nullptr;
int timer_seconds = 0;
int64_t due_us = 0;
int asked_seconds = 0;
int64_t asked_us = 0;
int idle_retries = IDLE_RETRIES;

std::string Spoken(int seconds) {
    if (seconds % 60 == 0) {
        return std::to_string(seconds / 60) + "分钟";
    }
    return std::to_string(seconds) + "秒";
}

// Runs in the main task, says the reminder once: the model answers this text. Keep
// it short, the cloud refuses long ones ("Detect is only for wake words"). After
// "start a timer" it usually is still
// listening, and it only takes the text when idle: end that conversation first.
void Alarm() {
    auto& app = Application::GetInstance();
    bool idle = app.GetDeviceState() == kDeviceStateIdle;
    app.WakeWordInvoke("计时到了");  // not idle: this closes the channel or stops the speech
    if (!idle && idle_retries-- > 0) {
        esp_timer_start_once(timer, IDLE_RETRY_US);
    }
}

void OnTimer(void*) {
    Application::GetInstance().Schedule(Alarm);
}

void StopTask(void*) {
    auto http = Board::GetInstance().GetNetwork()->CreateHttp(1);
    http->SetHeader("Content-Type", "application/json");
    http->SetContent("{\"action\":\"stop\"}");
    if (http->Open("POST", PI_CMD_URL)) {
        ESP_LOGI(TAG, "stop sent, status %d", http->GetStatusCode());
        http->Close();
    } else {
        ESP_LOGW(TAG, "stop not delivered");
    }
    vTaskDelete(nullptr);
}

}  // namespace

void StopDuck() {
    xTaskCreate(StopTask, "duck_stop", 6144, nullptr, 5, nullptr);
}

void AddTools() {
    esp_timer_create_args_t args = {};
    args.callback = OnTimer;
    args.name = "duck_timer";
    ESP_ERROR_CHECK(esp_timer_create(&args, &timer));

    auto& mcp = McpServer::GetInstance();
    mcp.AddTool("self.timer.start",
        "The user asked for a timer or a reminder after some time. This does not start it: "
        "it only returns a question, say that question to the user and wait for the answer.",
        PropertyList({Property("seconds", kPropertyTypeInteger, 1, MAX_TIMER_S)}),
        [](const PropertyList& properties) -> ReturnValue {
            asked_seconds = properties["seconds"].value<int>();
            asked_us = esp_timer_get_time();
            return "确认计时" + Spoken(asked_seconds) + "吗？请说确认。";
        });
    mcp.AddTool("self.timer.confirm",
        "Starts the timer. Call it only after asking with self.timer.start and only when the user "
        "clearly answered yes, such as 确认 or 对. If the answer is unclear or no, do not call it: no timer runs.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            int64_t waited = esp_timer_get_time() - asked_us;
            if (asked_seconds == 0 || waited < CONFIRM_MIN_US || waited > CONFIRM_MAX_US) {
                asked_seconds = 0;
                return std::string("not started, ask the user again");
            }
            timer_seconds = asked_seconds;
            asked_seconds = 0;
            idle_retries = IDLE_RETRIES;
            due_us = esp_timer_get_time() + timer_seconds * 1000000LL;
            esp_timer_stop(timer);
            esp_timer_start_once(timer, timer_seconds * 1000000LL);
            return "好的，开始计时" + Spoken(timer_seconds);
        });
    mcp.AddTool("self.timer.status", "Seconds left on the countdown timer, 0 when none is running.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            return static_cast<int>(std::max<int64_t>((due_us - esp_timer_get_time()) / 1000000LL, 0));
        });
    mcp.AddTool("self.timer.cancel",
        "Cancel the countdown timer.",
        PropertyList(),
        [](const PropertyList&) -> ReturnValue {
            esp_timer_stop(timer);
            due_us = 0;
            return true;
        });
}

}  // namespace duck_local
