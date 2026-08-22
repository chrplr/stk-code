//
//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2026 SuperTuxKart-Team
//
//  This program is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public License
//  as published by the Free Software Foundation; either version 3
//  of the License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

#include "gym/gym_json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace GymJson
{

/** Nesting depth limit. The protocol needs three levels (response > karts array
 *  > kart object); anything far beyond that is a client sending us something we
 *  do not want to recurse into. */
static const int MAX_DEPTH = 32;

// ============================================================================
// Value
// ============================================================================
const Value& Value::get(const std::string &key) const
{
    static const Value null_value;
    if (m_type != T_OBJECT) return null_value;
    std::map<std::string, Value>::const_iterator it = m_object.find(key);
    if (it == m_object.end()) return null_value;
    return it->second;
}   // get

//-----------------------------------------------------------------------------
double Value::asNumber(double d) const
{
    return m_type == T_NUMBER ? m_number : d;
}   // asNumber

//-----------------------------------------------------------------------------
int Value::asInt(int i) const
{
    return m_type == T_NUMBER ? (int)m_number : i;
}   // asInt

//-----------------------------------------------------------------------------
bool Value::asBool(bool b) const
{
    // A number is accepted as a boolean so that a client may send 0/1 for the
    // button fields; anything else keeps the default.
    if (m_type == T_BOOL)   return m_bool;
    if (m_type == T_NUMBER) return m_number != 0.0;
    return b;
}   // asBool

//-----------------------------------------------------------------------------
std::string Value::asString(const std::string &s) const
{
    return m_type == T_STRING ? m_string : s;
}   // asString

// ============================================================================
// Parser
// ============================================================================
namespace
{
    class Parser
    {
    private:
        const std::string &m_text;
        size_t             m_pos;
        std::string        m_error;

    public:
        Parser(const std::string &text) : m_text(text), m_pos(0) {}

        const std::string& getError() const { return m_error; }

        //---------------------------------------------------------------------
        void skipWhitespace()
        {
            while (m_pos < m_text.size())
            {
                const char c = m_text[m_pos];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                    m_pos++;
                else
                    break;
            }
        }   // skipWhitespace

        //---------------------------------------------------------------------
        bool fail(const std::string &message)
        {
            if (m_error.empty())
            {
                char buffer[32];
                sprintf(buffer, " at offset %d", (int)m_pos);
                m_error = message + buffer;
            }
            return false;
        }   // fail

        //---------------------------------------------------------------------
        bool atEnd()
        {
            skipWhitespace();
            return m_pos >= m_text.size();
        }   // atEnd

        //---------------------------------------------------------------------
        /** Parses a JSON string literal, resolving the escapes the protocol can
         *  actually produce. \\u is decoded to UTF-8 for the basic plane;
         *  surrogate pairs are combined. */
        bool parseString(std::string *out)
        {
            if (m_pos >= m_text.size() || m_text[m_pos] != '"')
                return fail("expected a string");
            m_pos++;
            out->clear();
            while (m_pos < m_text.size())
            {
                const char c = m_text[m_pos++];
                if (c == '"') return true;
                if (c != '\\')
                {
                    // Control characters are not legal unescaped in a string.
                    if ((unsigned char)c < 0x20)
                        return fail("unescaped control character in string");
                    out->push_back(c);
                    continue;
                }
                if (m_pos >= m_text.size())
                    return fail("truncated escape sequence");
                const char e = m_text[m_pos++];
                switch (e)
                {
                case '"' : out->push_back('"');  break;
                case '\\': out->push_back('\\'); break;
                case '/' : out->push_back('/');  break;
                case 'b' : out->push_back('\b'); break;
                case 'f' : out->push_back('\f'); break;
                case 'n' : out->push_back('\n'); break;
                case 'r' : out->push_back('\r'); break;
                case 't' : out->push_back('\t'); break;
                case 'u' :
                {
                    unsigned int cp = 0;
                    if (!parseHex4(&cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF)
                    {
                        // High surrogate: a low surrogate must follow.
                        if (m_pos + 1 < m_text.size() &&
                            m_text[m_pos] == '\\' && m_text[m_pos+1] == 'u')
                        {
                            m_pos += 2;
                            unsigned int low = 0;
                            if (!parseHex4(&low)) return false;
                            if (low < 0xDC00 || low > 0xDFFF)
                                return fail("invalid low surrogate");
                            cp = 0x10000 + ((cp - 0xD800) << 10)
                                         +  (low - 0xDC00);
                        }
                        else
                            return fail("unpaired high surrogate");
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: return fail("unknown escape sequence");
                }   // switch
            }   // while
            return fail("unterminated string");
        }   // parseString

        //---------------------------------------------------------------------
        bool parseHex4(unsigned int *out)
        {
            if (m_pos + 4 > m_text.size())
                return fail("truncated \\u escape");
            unsigned int value = 0;
            for (int i = 0; i < 4; i++)
            {
                const char c = m_text[m_pos++];
                value <<= 4;
                if      (c >= '0' && c <= '9') value |= (unsigned int)(c - '0');
                else if (c >= 'a' && c <= 'f') value |= (unsigned int)(c-'a'+10);
                else if (c >= 'A' && c <= 'F') value |= (unsigned int)(c-'A'+10);
                else return fail("invalid hex digit in \\u escape");
            }
            *out = value;
            return true;
        }   // parseHex4

        //---------------------------------------------------------------------
        static void appendUtf8(std::string *out, unsigned int cp)
        {
            if (cp < 0x80)
                out->push_back((char)cp);
            else if (cp < 0x800)
            {
                out->push_back((char)(0xC0 | (cp >> 6)));
                out->push_back((char)(0x80 | (cp & 0x3F)));
            }
            else if (cp < 0x10000)
            {
                out->push_back((char)(0xE0 | (cp >> 12)));
                out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                out->push_back((char)(0x80 | (cp & 0x3F)));
            }
            else
            {
                out->push_back((char)(0xF0 | (cp >> 18)));
                out->push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
                out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                out->push_back((char)(0x80 | (cp & 0x3F)));
            }
        }   // appendUtf8

        //---------------------------------------------------------------------
        bool parseNumber(Value *out)
        {
            const size_t start = m_pos;
            if (m_pos < m_text.size() && m_text[m_pos] == '-') m_pos++;
            const size_t int_start = m_pos;
            while (m_pos < m_text.size() &&
                   m_text[m_pos] >= '0' && m_text[m_pos] <= '9') m_pos++;
            if (m_pos == int_start) return fail("expected a digit");
            // JSON forbids leading zeros. Accepting them would silently read
            // 010 as 10, so reject rather than guess what was meant.
            if (m_text[int_start] == '0' && m_pos - int_start > 1)
                return fail("number has a leading zero");
            if (m_pos < m_text.size() && m_text[m_pos] == '.')
            {
                m_pos++;
                const size_t frac_start = m_pos;
                while (m_pos < m_text.size() &&
                       m_text[m_pos] >= '0' && m_text[m_pos] <= '9') m_pos++;
                if (m_pos == frac_start)
                    return fail("expected a digit after the decimal point");
            }
            if (m_pos < m_text.size() &&
                (m_text[m_pos] == 'e' || m_text[m_pos] == 'E'))
            {
                m_pos++;
                if (m_pos < m_text.size() &&
                    (m_text[m_pos] == '+' || m_text[m_pos] == '-')) m_pos++;
                const size_t exp_start = m_pos;
                while (m_pos < m_text.size() &&
                       m_text[m_pos] >= '0' && m_text[m_pos] <= '9') m_pos++;
                if (m_pos == exp_start)
                    return fail("expected a digit in the exponent");
            }
            // strtod is locale dependent for the decimal separator, but STK
            // never calls setlocale, so the C locale is in force here.
            const std::string number = m_text.substr(start, m_pos - start);
            out->setNumber(strtod(number.c_str(), NULL));
            return true;
        }   // parseNumber

        //---------------------------------------------------------------------
        bool parseLiteral(const char *literal, size_t length)
        {
            if (m_text.compare(m_pos, length, literal) != 0)
                return fail("invalid literal");
            m_pos += length;
            return true;
        }   // parseLiteral

        //---------------------------------------------------------------------
        bool parseValue(Value *out, int depth)
        {
            if (depth > MAX_DEPTH) return fail("nesting is too deep");
            skipWhitespace();
            if (m_pos >= m_text.size()) return fail("unexpected end of input");

            const char c = m_text[m_pos];
            switch (c)
            {
            case '{': return parseObject(out, depth);
            case '[': return parseArray(out, depth);
            case '"':
            {
                std::string s;
                if (!parseString(&s)) return false;
                out->setString(s);
                return true;
            }
            case 't':
                if (!parseLiteral("true", 4)) return false;
                out->setBool(true);
                return true;
            case 'f':
                if (!parseLiteral("false", 5)) return false;
                out->setBool(false);
                return true;
            case 'n':
                if (!parseLiteral("null", 4)) return false;
                out->setNull();
                return true;
            default:
                if (c == '-' || (c >= '0' && c <= '9'))
                    return parseNumber(out);
                return fail("unexpected character");
            }   // switch
        }   // parseValue

        //---------------------------------------------------------------------
        bool parseObject(Value *out, int depth)
        {
            m_pos++;   // consume '{'
            out->setObject();
            skipWhitespace();
            if (m_pos < m_text.size() && m_text[m_pos] == '}')
            {
                m_pos++;
                return true;
            }
            while (true)
            {
                skipWhitespace();
                std::string key;
                if (!parseString(&key)) return false;
                skipWhitespace();
                if (m_pos >= m_text.size() || m_text[m_pos] != ':')
                    return fail("expected ':'");
                m_pos++;
                Value member;
                if (!parseValue(&member, depth + 1)) return false;
                out->set(key, member);
                skipWhitespace();
                if (m_pos >= m_text.size()) return fail("unterminated object");
                if (m_text[m_pos] == ',') { m_pos++; continue; }
                if (m_text[m_pos] == '}') { m_pos++; return true; }
                return fail("expected ',' or '}'");
            }
        }   // parseObject

        //---------------------------------------------------------------------
        bool parseArray(Value *out, int depth)
        {
            m_pos++;   // consume '['
            out->setArray();
            skipWhitespace();
            if (m_pos < m_text.size() && m_text[m_pos] == ']')
            {
                m_pos++;
                return true;
            }
            while (true)
            {
                Value element;
                if (!parseValue(&element, depth + 1)) return false;
                out->push(element);
                skipWhitespace();
                if (m_pos >= m_text.size()) return fail("unterminated array");
                if (m_text[m_pos] == ',') { m_pos++; continue; }
                if (m_text[m_pos] == ']') { m_pos++; return true; }
                return fail("expected ',' or ']'");
            }
        }   // parseArray
    };   // class Parser
}   // anonymous namespace

//-----------------------------------------------------------------------------
bool parse(const std::string &text, Value *out, std::string *error)
{
    Parser parser(text);
    *out = Value();
    if (!parser.parseValue(out, 0))
    {
        if (error) *error = parser.getError();
        return false;
    }
    // Trailing garbage is an error rather than something to ignore: a line that
    // is two concatenated objects means the client's framing has broken, and
    // silently taking the first one would hide that until much later.
    if (!parser.atEnd())
    {
        if (error) *error = "trailing data after the JSON value";
        return false;
    }
    return true;
}   // parse

// ============================================================================
// Writer
// ============================================================================
void Writer::comma()
{
    if (m_first.back())
        m_first.back() = false;
    else
        m_data += ",";
}   // comma

//-----------------------------------------------------------------------------
void Writer::addNull(const std::string &key)
{
    comma();
    m_data += quote(key) + ":null";
}   // addNull

//-----------------------------------------------------------------------------
void Writer::addBool(const std::string &key, bool value)
{
    comma();
    m_data += quote(key) + (value ? ":true" : ":false");
}   // addBool

//-----------------------------------------------------------------------------
void Writer::addInt(const std::string &key, long long value)
{
    comma();
    char buffer[32];
    sprintf(buffer, "%lld", value);
    m_data += quote(key) + ":" + buffer;
}   // addInt

//-----------------------------------------------------------------------------
/** Writes a number with %.6g. JSON has no notion of NaN or infinity, and a
 *  literal NaN would make the client's json.loads raise where it cannot
 *  recover, so a non-finite value is written as null instead. */
void Writer::addFloat(const std::string &key, double value)
{
    comma();
    m_data += quote(key) + ":";
    if (value != value || value > 1e300 || value < -1e300)
    {
        m_data += "null";
        return;
    }
    char buffer[64];
    sprintf(buffer, "%.6g", value);
    m_data += buffer;
}   // addFloat

//-----------------------------------------------------------------------------
void Writer::addString(const std::string &key, const std::string &value)
{
    comma();
    m_data += quote(key) + ":" + quote(value);
}   // addString

//-----------------------------------------------------------------------------
void Writer::addFloatArray(const std::string &key,
                           const std::vector<double> &values)
{
    comma();
    m_data += quote(key) + ":[";
    for (unsigned int i = 0; i < values.size(); i++)
    {
        if (i > 0) m_data += ",";
        const double v = values[i];
        if (v != v || v > 1e300 || v < -1e300)
        {
            m_data += "null";
            continue;
        }
        char buffer[64];
        sprintf(buffer, "%.6g", v);
        m_data += buffer;
    }
    m_data += "]";
}   // addFloatArray

//-----------------------------------------------------------------------------
void Writer::beginObject(const std::string &key)
{
    comma();
    m_data += quote(key) + ":{";
    m_first.push_back(true);
}   // beginObject

//-----------------------------------------------------------------------------
void Writer::endObject()
{
    m_data += "}";
    m_first.pop_back();
}   // endObject

//-----------------------------------------------------------------------------
void Writer::beginArray(const std::string &key)
{
    comma();
    m_data += quote(key) + ":[";
    m_first.push_back(true);
}   // beginArray

//-----------------------------------------------------------------------------
void Writer::endArray()
{
    m_data += "]";
    m_first.pop_back();
}   // endArray

//-----------------------------------------------------------------------------
void Writer::beginArrayElement()
{
    comma();
    m_data += "{";
    m_first.push_back(true);
}   // beginArrayElement

//-----------------------------------------------------------------------------
void Writer::endArrayElement()
{
    m_data += "}";
    m_first.pop_back();
}   // endArrayElement

//-----------------------------------------------------------------------------
std::string Writer::toString()
{
    return m_data + "}";
}   // toString

//-----------------------------------------------------------------------------
std::string Writer::quote(const std::string &s)
{
    std::string result = "\"";
    for (unsigned int i = 0; i < s.size(); i++)
    {
        const unsigned char c = (unsigned char)s[i];
        switch (c)
        {
        case '"' : result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b";  break;
        case '\f': result += "\\f";  break;
        case '\n': result += "\\n";  break;
        case '\r': result += "\\r";  break;
        case '\t': result += "\\t";  break;
        default:
            if (c < 0x20)
            {
                char buffer[8];
                sprintf(buffer, "\\u%04x", (unsigned int)c);
                result += buffer;
            }
            else
                result += (char)c;
        }   // switch
    }
    return result + "\"";
}   // quote

}   // namespace GymJson
