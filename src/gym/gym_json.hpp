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

#ifndef HEADER_GYM_JSON_HPP
#define HEADER_GYM_JSON_HPP

#include <map>
#include <string>
#include <vector>

/** A deliberately small JSON reader and writer, sized for the gym protocol and
 *  nothing else.
 *
 *  STK has no JSON parser: HardwareStats::Json (config/hardware_stats.hpp) only
 *  writes. The gym protocol needs to read too, and it is small and entirely
 *  under our control - flat-ish objects of numbers, strings, bools and short
 *  arrays - so ~250 lines here is preferred over vendoring a large third party
 *  header into the game.
 *
 *  The parser never throws and never aborts: a malformed line must come back as
 *  a "bad_json" error response, because a client bug must not take the server
 *  down.
 */
namespace GymJson
{
    // ------------------------------------------------------------------------
    /** A parsed JSON value. Deliberately a plain tagged struct rather than a
     *  variant, so that it compiles everywhere STK does. */
    class Value
    {
    public:
        enum Type { T_NULL, T_BOOL, T_NUMBER, T_STRING, T_ARRAY, T_OBJECT };

    private:
        Type                          m_type;
        bool                          m_bool;
        double                        m_number;
        std::string                   m_string;
        std::vector<Value>            m_array;
        std::map<std::string, Value>  m_object;

    public:
        Value() : m_type(T_NULL), m_bool(false), m_number(0.0) {}

        Type getType()     const { return m_type;              }
        bool isNull()      const { return m_type == T_NULL;    }
        bool isBool()      const { return m_type == T_BOOL;    }
        bool isNumber()    const { return m_type == T_NUMBER;  }
        bool isString()    const { return m_type == T_STRING;  }
        bool isArray()     const { return m_type == T_ARRAY;   }
        bool isObject()    const { return m_type == T_OBJECT;  }

        void setNull()                     { *this = Value();                  }
        void setBool(bool b)               { m_type = T_BOOL;   m_bool   = b;  }
        void setNumber(double d)           { m_type = T_NUMBER; m_number = d;  }
        void setString(const std::string&s){ m_type = T_STRING; m_string = s;  }
        void setArray()                    { m_type = T_ARRAY;                 }
        void setObject()                   { m_type = T_OBJECT;                }

        void push(const Value &v)          { m_array.push_back(v);             }
        void set(const std::string &k, const Value &v) { m_object[k] = v;      }

        const std::vector<Value>& getArray() const { return m_array;  }
        const std::map<std::string, Value>& getObject() const
                                                   { return m_object; }

        /** Returns true if this is an object with the given key. */
        bool has(const std::string &key) const
        {
            return m_type == T_OBJECT && m_object.find(key) != m_object.end();
        }
        /** Returns the member, or a null Value if absent. Never fails, so
         *  reading an optional field needs no special case at the call site. */
        const Value& get(const std::string &key) const;

        // Typed accessors with a default. They do not convert between types:
        // asking for a number and getting a string yields the default, which is
        // what makes a wrong-typed field show up as a bad_action rather than as
        // a silently zeroed control.
        double      asNumber(double d = 0.0)              const;
        int         asInt   (int i = 0)                   const;
        bool        asBool  (bool b = false)              const;
        std::string asString(const std::string &s = "")   const;
    };   // class Value

    // ------------------------------------------------------------------------
    /** Parses one complete JSON value. Returns false and fills \p error on any
     *  malformed input, including trailing garbage. */
    bool parse(const std::string &text, Value *out, std::string *error);

    // ------------------------------------------------------------------------
    /** Builds a compact single-line JSON object.
     *
     *  Written as an explicit builder rather than by serialising a Value,
     *  because every response is produced once, in order, and this keeps the
     *  state extraction free of intermediate allocations. */
    class Writer
    {
    private:
        std::string m_data;
        /** One flag per open object/array, true until that level's first
         *  member is written, so commas separate members instead of preceding
         *  the first one. A stack rather than a single flag because nested
         *  objects (the per-kart rows) reopen the question at each level. */
        std::vector<bool> m_first;

        void comma();

    public:
        Writer() { m_data = "{"; m_first.push_back(true); }

        void addNull  (const std::string &key);
        void addBool  (const std::string &key, bool value);
        void addInt   (const std::string &key, long long value);
        void addFloat (const std::string &key, double value);
        void addString(const std::string &key, const std::string &value);

        /** Adds an array of numbers, e.g. a position or a lookahead row. */
        void addFloatArray(const std::string &key,
                           const std::vector<double> &values);

        /** Opens a nested object or array; the caller must close it. Nested
         *  objects are written by beginObject/endObject pairs so that the
         *  state builder can stream sub-structures without a second Writer. */
        void beginObject(const std::string &key);
        void endObject();
        void beginArray(const std::string &key);
        void endArray();
        /** Opens an unnamed object inside an array. */
        void beginArrayElement();
        void endArrayElement();

        /** Closes the top level object and returns the finished line. */
        std::string toString();

        /** Escapes a string as a JSON string literal, quotes included. */
        static std::string quote(const std::string &s);
    };   // class Writer

}   // namespace GymJson

#endif
