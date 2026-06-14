#pragma once
#include <string>
#include <vector>
#include <regex>
#include <algorithm>
#include <iostream>

namespace lora_kernel {

class TextSanitizer {
public:
    std::string sanitize(const std::string& input) const {
        std::string out;
        out.reserve(input.size());
        for (unsigned char c : input) {
            if (c >= 0x20 || c == '\n' || c == '\r' || c == '\t')
                out += static_cast<char>(c);
            else
                out += ' ';
        }
        if (out.size() > 100000) out.resize(100000);
        return out;
    }

    bool contains_sql_patterns(const std::string& text) const {
        static const char* const pats[] = {
            "DROP ", "DELETE ", "INSERT ", "UPDATE ", "SELECT ",
            "UNION ", "EXEC ", "XP_", "--", "/*", "*/", ";",
            "OR 1=1", "AND 1=1", nullptr
        };
        std::string upper(text.size(), '\0');
        std::transform(text.begin(), text.end(), upper.begin(), ::toupper);
        for (int i = 0; pats[i]; ++i)
            if (upper.find(pats[i]) != std::string::npos) return true;
        return false;
    }
};

class PIIMaskingFilter {
private:
    const std::regex email_{R"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})"};
    const std::regex phone_{R"(\+?[0-9]{1,3}[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})"};
    const std::regex ssn_  {R"(\b\d{3}-\d{2}-\d{4}\b)"};
    const std::regex cc_   {R"(\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b)"};

public:
    std::string mask(const std::string& text) const {
        std::string out = std::regex_replace(text, email_, "[EMAIL]");
        out = std::regex_replace(out, phone_, "[PHONE]");
        out = std::regex_replace(out, ssn_,   "[SSN]");
        out = std::regex_replace(out, cc_,    "[CC]");
        return out;
    }
    bool contains_pii(const std::string& text) const {
        return std::regex_search(text, email_) ||
               std::regex_search(text, phone_) ||
               std::regex_search(text, ssn_)   ||
               std::regex_search(text, cc_);
    }
};

} // namespace lora_kernel
