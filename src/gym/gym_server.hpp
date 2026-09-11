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

#ifndef HEADER_GYM_SERVER_HPP
#define HEADER_GYM_SERVER_HPP

#include <cstdio>
#include <string>

class AbstractKart;
class Controller;
namespace GymJson { class Value; }

/** Drives a race from an external process over a line based protocol.
 *
 *  Wire format: one JSON object per line, in each direction, request and
 *  response paired by id. It is meant to be readable and to be driveable by
 *  hand - start the binary and type {"id":1,"cmd":"hello"}.
 *
 *  The server reports facts and never a reward: the reward scheme, the
 *  termination rule and the observation tensor all live in the client, so they
 *  can be changed without rebuilding this binary.
 *
 *  Unlike the equivalent servers for simpler games, this one holds exactly one
 *  session. World, RaceManager, Track and the Bullet world are all process
 *  globals, so a second race cannot exist alongside the first. env_id is part
 *  of the schema anyway - so that the client code does not fork - but it must
 *  be 0, and a vector environment runs one child process per environment.
 *
 *  The track is fixed for the life of the process: loading one costs seconds,
 *  so reset restarts the race in place (RaceManager::rerunRace) and a different
 *  track means a different child.
 *
 *  --gym-human turns the server round: a person drives, through STK's own
 *  input system, and the game keeps its own clock, frame pacing and sound
 *  (MainLoop::run as usual). The protocol is then polled once per frame and
 *  only reports - state, plus the controls the kart applied - and restarts
 *  the race on reset. step is refused in this mode, so that a client cannot
 *  mistake a race that runs on its own for one it is stepping.
 */
class GymServer
{
private:
    static bool         m_enabled;
    static bool         m_human;
    static bool         m_expert;
    static int          m_frame_skip;
    static int          m_lookahead_k;
    static bool         m_include_karts;

    /** The real stdout, saved before file descriptor 1 is pointed at stderr.
     *  Everything STK, irrlicht or a bundled library prints then lands on
     *  stderr, and this descriptor carries the protocol and nothing else. */
    static FILE        *m_protocol_out;

    /** World id of the kart the agent drives. */
    unsigned int        m_kart_id;
    /** False until the first reset, so that stepping before resetting is an
     *  error the client can recognise rather than an undefined observation. */
    bool                m_has_reset;
    bool                m_quit;
    /** Bytes read from stdin in human mode that do not yet end a line. */
    std::string         m_pending;
    bool                m_kart_found;

    std::string  handle(const std::string &line);
    std::string  error(int id, const char *kind, const std::string &message);
    std::string  buildState(int id);
    std::string  buildStateBatch(int id);
    bool         checkEnvId(const GymJson::Value &request, int *id_out,
                            std::string *response);
    void         findKart();
    void         applyAction(const GymJson::Value &action);
    void         resetRace();
    void         advanceToRacePhase();
    void         stepTicks(int ticks);
    void         updateGraphics();

public:
                 GymServer();

    /** Turns gym mode on and moves the protocol off the noisy stdout. Called
     *  while the command line is first scanned, before anything has had a
     *  chance to print. */
    static void  enable();
    static bool  isEnabled()            { return m_enabled;       }
    static void  setHuman(bool h);
    static bool  isHuman()              { return m_human;         }
    static void  setExpert(bool e)      { m_expert = e;           }
    static bool  isExpert()             { return m_expert;        }
    static void  setFrameSkip(int n)    { m_frame_skip = n;       }
    static void  setLookahead(int n)    { m_lookahead_k = n;      }
    static void  setIncludeKarts(bool b){ m_include_karts = b;    }

    /** Builds the controller for the kart in the player slot. Called from
     *  World::createKart so that all the gym specific knowledge stays here. */
    static Controller* createPlayerController(AbstractKart *kart);

    /** Reads commands until stdin reaches end of file, which is how the process
     *  exits when its parent dies. Replaces MainLoop::run in gym mode. */
    void         run();
    /** Human mode: answers whatever commands have arrived, without blocking.
     *  Called by MainLoop::run once per frame. Stdin at end of file requests
     *  the loop to stop, as run() does by returning. */
    void         pollOnce();
    /** The server MainLoop polls in human mode, created on first use. */
    static GymServer* human();
};   // class GymServer

#endif
