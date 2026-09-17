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
#include <vector>

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
 *
 *  A request may add "frame": true, and the response then describes the
 *  rendered frame - {"width":W,"height":H,"format":"rgb8"} - and is followed,
 *  after its newline, by W*H*3 raw bytes: rows top to bottom, RGB. Pixels do
 *  not fit a line-based text format, and encoding them as text would cost more
 *  than the render itself at the sizes a screen needs, so the one exception to
 *  "one JSON object per line" is a binary block whose length the line before
 *  it states. --gym-hidden keeps the window off the screen for the client
 *  that shows the frames itself.
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
    static bool         m_hidden;
    static bool         m_keys;
    /** Set for the duration of one render so that the renderer's hook,
     *  which runs for every frame drawn, knows this is the one to keep. The
     *  frame itself is static too: there is one server per process, and the
     *  hook has no instance to reach. */
    static bool         m_capture_wanted;
    static std::vector<unsigned char> m_frame;
    static unsigned int m_frame_width;
    static unsigned int m_frame_height;
    static bool         m_frame_ready;

    /** The framebuffer object the frame is drawn into while the window is
     *  hidden, or 0 when the window is on the screen and its own back buffer
     *  is what gets drawn and read.
     *
     *  A window that is never mapped has no usable default framebuffer: X11
     *  leaves its contents undefined, so the readback returns whatever happens
     *  to be on the screen at those coordinates -- the desktop, or black --
     *  while the game reports a race running correctly. Some drivers do keep a
     *  valid buffer, which is why this went unnoticed at first; none of them
     *  promise to. A framebuffer object is defined everywhere, mapped window or
     *  not, so in hidden mode this is what IrrDriver::getDefaultFramebuffer()
     *  hands to everything that draws "to the screen", and what captureFrame()
     *  reads back. The colour attachment is a texture and the depth/stencil a
     *  renderbuffer, sized to the window and rebuilt if that size changes. */
    static unsigned int m_offscreen_fbo;
    static unsigned int m_offscreen_color;
    static unsigned int m_offscreen_depth;
    static unsigned int m_offscreen_width;
    static unsigned int m_offscreen_height;

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
    bool         frameRefused(int id, std::string *response);
    void         renderFrame();
    static void  ensureOffscreen();
    static void  releaseOffscreen();
    void         writeResponse(const std::string &response);
    bool         checkEnvId(const GymJson::Value &request, int *id_out,
                            std::string *response);
    void         findKart();
    void         applyAction(const GymJson::Value &action);
    void         resetRace();
    void         advanceToRacePhase();
    void         stepTicks(int ticks);
    void         updateGraphics();
    bool         wantsFrame(const GymJson::Value &request);

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
    static void  setHidden(bool h)      { if (h) enable(); m_hidden = h; }
    static bool  isHidden()             { return m_hidden;        }
    static void  setKeys(bool k)        { m_keys = k;             }
    static bool  isKeys()               { return m_keys;          }

    /** Reads the frame back if a request asked for one. Called by the
     *  renderers right before the buffers are swapped, which is the one point
     *  where the frame is complete, HUD included. */
    static void  captureFrame();

    /** The framebuffer everything should draw into, or 0 for the window's own.
     *  Non-zero only in hidden mode, where the window's default framebuffer
     *  cannot be read back (see m_offscreen_fbo); IrrDriver::
     *  getDefaultFramebuffer() returns this in its place. */
    static unsigned int offscreenFramebuffer() { return m_offscreen_fbo; }

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
