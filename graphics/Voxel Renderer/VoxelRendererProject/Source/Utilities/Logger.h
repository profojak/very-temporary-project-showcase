#pragma once

#include <chrono>
#include <iostream>
#include <mutex>

class Logger {
public:
    enum class Level {
        DEBUG,
        INFO,
        WARNING,
        FATAL
    };

    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLevel(Level level) {
        m_level = level;
    }

    template<typename... Args>
    void debug(Args... args) {
        log(Level::DEBUG, args...);
    }

    template<typename... Args>
    void info(Args... args) {
        log(Level::INFO, args...);
    }

    template<typename... Args>
    void warning(Args... args) {
        log(Level::WARNING, args...);
    }

    template<typename... Args>
    void fatal(Args... args) {
        log(Level::FATAL, args...);
    }

private:
    Logger() : m_level(Level::INFO) {}
    ~Logger() = default;

    std::string levelToString(Level level) {
        switch (level) {
        case Level::DEBUG:   return "DEBUG";
        case Level::INFO:    return "INFO";
        case Level::WARNING: return "WARNING";
        case Level::FATAL:   return "FATAL";
        default:             return "UNKNOWN";
        }
    }

    std::string getCurrentTime() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::tm timeInfo;
        localtime_s(&timeInfo, &time);

        std::stringstream ss;
        ss << std::put_time(&timeInfo, "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    template<typename T>
    void formatArg(std::stringstream& ss, T&& arg) {
        ss << std::forward<T>(arg);
    }

    template<typename T, typename... Args>
    void formatArg(std::stringstream& ss, T&& arg, Args&&... args) {
        ss << std::forward<T>(arg);
        formatArg(ss, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void log(Level level, Args... args) {
        if (level < m_level) return;

        std::lock_guard<std::mutex> lock(m_mutex);

        std::stringstream message;
        message << "[" << getCurrentTime() << "] [" << levelToString(level) << "] ";
        formatArg(message, args...);
        message << std::endl;

        std::cout << message.str();
    }

    Level m_level;
    std::mutex m_mutex;
};

#define LOG_DEBUG(...) Logger::getInstance().debug(__VA_ARGS__)
#define LOG_INFO(...) Logger::getInstance().info(__VA_ARGS__)
#define LOG_WARNING(...) Logger::getInstance().warning(__VA_ARGS__)
#define LOG_FATAL(...) Logger::getInstance().fatal(__VA_ARGS__)
