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

#include "gym/gym_server.hpp"

#include "config/stk_config.hpp"
#include "config/user_config.hpp"
#include "graphics/camera/camera.hpp"
#include "graphics/irr_driver.hpp"
#include "guiengine/engine.hpp"
#include "gym/gym_controller.hpp"
#include "gym/gym_json.hpp"
#include "gym/gym_state.hpp"
#include "karts/abstract_kart.hpp"
#include "karts/controller/local_player_controller.hpp"
#include "main_loop.hpp"
#include "modes/linear_world.hpp"
#include "modes/world.hpp"
#include "race/race_manager.hpp"
#include "utils/log.hpp"
#include "utils/random_generator.hpp"

#include <cstdlib>
#include <iostream>

#ifndef SERVER_ONLY
#  include <IrrlichtDevice.h>
#endif

#ifdef WIN32
#  include <io.h>
#  include <windows.h>
#  define STK_DUP  _dup
#  define STK_DUP2 _dup2
#  define STK_FDOPEN _fdopen
#else
#  include <poll.h>
#  include <unistd.h>
#  define STK_DUP  dup
#  define STK_DUP2 dup2
#  define STK_FDOPEN fdopen
#endif

bool  GymServer::m_enabled       = false;
bool  GymServer::m_human         = false;
bool  GymServer::m_expert        = false;
int   GymServer::m_frame_skip    = 6;
int   GymServer::m_lookahead_k   = 5;
bool  GymServer::m_include_karts = true;
FILE *GymServer::m_protocol_out  = NULL;

/** Bumped whenever the wire format changes in a way a client must notice. The
 *  client checks it at the handshake and refuses to run on a mismatch. */
static const int PROTOCOL_VERSION = 1;

/** Error kinds. The client branches on these and never on the prose message. */
static const char *ERR_BAD_JSON     = "bad_json";
static const char *ERR_UNKNOWN_CMD  = "unknown_cmd";
static const char *ERR_NO_SUCH_ENV  = "no_such_env";
static const char *ERR_BAD_ACTION   = "bad_action";
static const char *ERR_NOT_RESET    = "not_reset";
static const char *ERR_BAD_BATCH    = "bad_batch";
static const char *ERR_NOT_SUPPORTED= "not_supported";

//-----------------------------------------------------------------------------
GymServer::GymServer()
{
    m_kart_id    = 0;
    m_has_reset  = false;
    m_quit       = false;
    m_kart_found = false;
}   // GymServer

//-----------------------------------------------------------------------------
/** Human mode implies gym mode. The race keeps its ready-set-go countdown - it
 *  is part of what the person experiences - unless -R is given, which STK's
 *  own command line parsing applies after this. */
void GymServer::setHuman(bool h)
{
    if (h) enable();
    m_human = h;
    if (h) UserConfigParams::m_race_now = false;
}   // setHuman

//-----------------------------------------------------------------------------
GymServer* GymServer::human()
{
    static GymServer server;
    return &server;
}   // human

//-----------------------------------------------------------------------------
/** Claims the real stdout for the protocol and points file descriptor 1 at
 *  stderr.
 *
 *  STK's Log writes to stdout with printf, and so do irrlicht and several
 *  bundled libraries. Rather than teach each of them a new destination, the
 *  descriptor itself is redirected: from here on anything printed anywhere in
 *  the process lands on stderr, which the client inherits, and the saved
 *  descriptor carries the protocol and nothing else.
 */
void GymServer::enable()
{
    if (m_enabled) return;
    m_enabled = true;

    const int saved = STK_DUP(1);
    if (saved < 0)
    {
        // Without a private descriptor the protocol would be interleaved with
        // log output, which the client cannot parse. Better to say so and stop
        // than to produce a stream that fails mysteriously later.
        fprintf(stderr, "[gym] cannot duplicate stdout; refusing to start.\n");
        exit(1);
    }
    STK_DUP2(2, 1);
    m_protocol_out = STK_FDOPEN(saved, "w");
    if (m_protocol_out == NULL)
    {
        fprintf(stderr, "[gym] cannot reopen the saved stdout; refusing to "
                        "start.\n");
        exit(1);
    }

    // The race must start without menus and without the ready-set-go countdown,
    // the same way -R does it.
    UserConfigParams::m_no_start_screen = true;
    UserConfigParams::m_race_now        = true;
}   // enable

//-----------------------------------------------------------------------------
Controller* GymServer::createPlayerController(AbstractKart *kart)
{
    if (m_human)
    {
        // The seat is a person's: STK's own controller, reading the keyboard
        // or gamepad that -N assigned to the active player, with the camera
        // and sound a local player gets. The server only watches.
        return new LocalPlayerController(kart, 0/*local_player_id*/,
                                         HANDICAP_NONE);
    }
    if (m_expert)
    {
        // The reference policy: STK's own racing AI in the agent's seat, so
        // that a training run has a ceiling to be measured against. A learned
        // policy that does not approach it has not learned anything worth
        // having.
        if (!GUIEngine::isNoGraphics())
            Camera::createCamera(kart, 0);
        return new GymExpertController(kart);
    }
    return new GymController(kart);
}   // createPlayerController

//-----------------------------------------------------------------------------
/** Locates the kart in the player slot. RaceManager puts ghosts first, then the
 *  AI karts, then the players, but the order is not relied on here. */
void GymServer::findKart()
{
    m_kart_id = 0;
    World *world = World::getWorld();
    if (world == NULL) return;
    for (unsigned int i = 0; i < world->getNumKarts(); i++)
    {
        if (RaceManager::get()->getKartType(i) == RaceManager::KT_PLAYER)
        {
            m_kart_id = i;
            return;
        }
    }
    Log::warn("GymServer", "No player kart found; observing kart 0.");
}   // findKart

//-----------------------------------------------------------------------------
void GymServer::run()
{
    if (World::getWorld() == NULL)
    {
        fprintf(stderr, "[gym] no world was created; nothing to drive.\n");
        return;
    }
    findKart();

    std::string line;
    while (!m_quit && std::getline(std::cin, line))
    {
        // A trailing \r is what a client on Windows, or a hand typed line
        // pasted from a text file, will send.
        if (!line.empty() && line[line.size() - 1] == '\r')
            line.erase(line.size() - 1);
        if (line.empty()) continue;

        const std::string response = handle(line);
        fputs(response.c_str(), m_protocol_out);
        fputc('\n', m_protocol_out);
        // The client is blocked reading one line and will not send the next
        // request until it arrives.
        fflush(m_protocol_out);
    }
}   // run

//-----------------------------------------------------------------------------
/** Reads whatever stdin holds right now and answers each complete line.
 *
 *  Runs on the game's thread, between two frames, so World is safe to read
 *  and a request costs at most one frame of latency. Stdin at end of file is
 *  the parent process gone (or done), and the loop is asked to stop exactly
 *  as run() would return. */
void GymServer::pollOnce()
{
    if (m_quit) return;
    if (!m_kart_found && World::getWorld())
    {
        findKart();
        m_kart_found = true;
    }

    char buffer[4096];
    for (;;)
    {
#ifdef WIN32
        HANDLE in = (HANDLE)_get_osfhandle(0);
        DWORD available = 0;
        if (!PeekNamedPipe(in, NULL, 0, NULL, &available, NULL))
        {
            // Not a pipe (a console, say): fall back to a blocking read only
            // when the handle signals data, else give up polling this frame.
            if (WaitForSingleObject(in, 0) != WAIT_OBJECT_0) break;
            available = 1;
        }
        if (available == 0) break;
        const int n = _read(0, buffer, sizeof(buffer));
#else
        struct pollfd pfd;
        pfd.fd      = 0;
        pfd.events  = POLLIN;
        pfd.revents = 0;
        const int ready = poll(&pfd, 1, 0/*timeout ms*/);
        if (ready <= 0) break;
        const ssize_t n = read(0, buffer, sizeof(buffer));
#endif
        if (n < 0) break;
        if (n == 0)
        {
            // End of file: nobody will ever ask again.
            m_quit = true;
            if (main_loop) main_loop->requestAbort();
            return;
        }
        m_pending.append(buffer, (size_t)n);
        if ((size_t)n < sizeof(buffer)) break;
    }

    size_t start = 0;
    for (;;)
    {
        const size_t nl = m_pending.find('\n', start);
        if (nl == std::string::npos) break;
        std::string line = m_pending.substr(start, nl - start);
        start = nl + 1;
        if (!line.empty() && line[line.size() - 1] == '\r')
            line.erase(line.size() - 1);
        if (line.empty()) continue;
        const std::string response = handle(line);
        fputs(response.c_str(), m_protocol_out);
        fputc('\n', m_protocol_out);
        fflush(m_protocol_out);
        if (m_quit) break;
    }
    m_pending.erase(0, start);
}   // pollOnce

//-----------------------------------------------------------------------------
std::string GymServer::error(int id, const char *kind,
                             const std::string &message)
{
    GymJson::Writer writer;
    writer.addInt("id", id);
    writer.addBool("ok", false);
    writer.addString("kind", kind);
    writer.addString("error", message);
    return writer.toString();
}   // error

//-----------------------------------------------------------------------------
/** Checks the env_id of a request. There is exactly one session per process, so
 *  anything but 0 is a client that believes otherwise and should be told. */
bool GymServer::checkEnvId(const GymJson::Value &request, int *id_out,
                           std::string *response)
{
    const int id = request.get("id").asInt(0);
    *id_out = id;
    const GymJson::Value &env_id = request.get("env_id");
    if (env_id.isNull() || env_id.asInt(0) == 0) return true;
    *response = error(id, ERR_NO_SUCH_ENV,
                      "this build holds one environment per process; env_id "
                      "must be 0 and a vector environment runs one child "
                      "process per environment");
    return false;
}   // checkEnvId

//-----------------------------------------------------------------------------
std::string GymServer::buildState(int id)
{
    GymJson::Writer writer;
    writer.addInt("id", id);
    writer.addBool("ok", true);
    GymState::write(&writer, m_kart_id, (unsigned int)m_lookahead_k,
                    m_include_karts);
    return writer.toString();
}   // buildState

//-----------------------------------------------------------------------------
/** The batch form of a response: the same state, wrapped in a one element
 *  "states" array. Batches never hold more than one environment here, but the
 *  shape is kept so that the client's batch path needs no special case. */
std::string GymServer::buildStateBatch(int id)
{
    GymJson::Writer writer;
    writer.addInt("id", id);
    writer.addBool("ok", true);
    writer.beginArray("states");
    writer.beginArrayElement();
    GymState::writeFields(&writer, m_kart_id, (unsigned int)m_lookahead_k,
                          m_include_karts);
    writer.endArrayElement();
    writer.endArray();
    return writer.toString();
}   // buildStateBatch

//-----------------------------------------------------------------------------
/** Copies a step request's action onto the controller. Absent fields keep their
 *  neutral value, so a client that only wants to steer can send only "steer". */
void GymServer::applyAction(const GymJson::Value &action)
{
    World *world = World::getWorld();
    if (world == NULL) return;
    GymController *controller = dynamic_cast<GymController*>
                                    (world->getKart(m_kart_id)->getController());
    // In expert mode the seat is taken by SkiddingAI, and actions are ignored
    // on purpose: the client is watching the reference policy, not driving.
    if (controller == NULL) return;

    GymAction a;
    a.m_steer  = (float)action.get("steer" ).asNumber(0.0);
    a.m_accel  = (float)action.get("accel" ).asNumber(0.0);
    a.m_brake  = action.get("brake" ).asBool(false);
    a.m_nitro  = action.get("nitro" ).asBool(false);
    a.m_fire   = action.get("fire"  ).asBool(false);
    a.m_rescue = action.get("rescue").asBool(false);
    a.m_skid   = action.get("skid"  ).asInt(KartControl::SC_NONE);
    if (a.m_skid < KartControl::SC_NONE || a.m_skid > KartControl::SC_RIGHT)
        a.m_skid = KartControl::SC_NONE;
    // KartControl clamps steer and accel itself, so out of range values are
    // saturated rather than rejected: a squashed policy output should not
    // abort an episode.
    controller->setAction(a);
}   // applyAction

//-----------------------------------------------------------------------------
/** Restarts the race in place.
 *
 *  Track sectors are recomputed by LinearWorld::update, which has not run yet
 *  when the first observation of an episode is taken, so they are refreshed
 *  here; otherwise distance_down_track would open the episode holding the value
 *  it had at the end of the previous one.
 *
 *  A restart does not restore the physics world bit for bit: measured on
 *  hacienda, two episodes with the same seed and the same actions start about
 *  0.7 mm apart and drift to about 2 cm over 300 steps, and how much the world
 *  ran before the restart is what decides it. Episodes preceded by identical
 *  histories are identical, so a whole run from process start with a fixed seed
 *  and a fixed action sequence reproduces exactly; individual episodes within a
 *  run are not bit-identical to each other.
 */
void GymServer::resetRace()
{
    if (m_human)
    {
        // The live loop runs the countdown and refreshes the track sectors
        // on its next update; nothing is stepped here. m_race_now is left as
        // the command line set it.
        RaceManager::get()->rerunRace();
        return;
    }
    // WorldStatus clears this once the race is under way, so it is set again
    // for every restart; without it each episode would pay a ready-set-go
    // countdown.
    UserConfigParams::m_race_now = true;
    RaceManager::get()->rerunRace();
    advanceToRacePhase();

    LinearWorld *linear = dynamic_cast<LinearWorld*>(World::getWorld());
    if (linear) linear->updateTrackSectors();
}   // resetRace

//-----------------------------------------------------------------------------
/** Runs ticks until the race is actually under way. With m_race_now set this
 *  normally costs nothing, but a mode that insists on a countdown must not
 *  return an observation taken before the kart may move. */
void GymServer::advanceToRacePhase()
{
    const int max_ticks = stk_config->time2Ticks(30.0f);
    int ticks = 0;
    while (World::getWorld() && World::getWorld()->isStartPhase() &&
           ticks < max_ticks)
    {
        main_loop->updateSingleTick(false/*fast_forward*/, NULL);
        ticks++;
    }
}   // advanceToRacePhase

//-----------------------------------------------------------------------------
/** Advances the race clock by exactly \p ticks.
 *
 *  Loops on the clock rather than counting calls, because a tick that reloads
 *  the world runs the physics but returns before the clock is updated, which
 *  would otherwise make one step per episode shorter than all the others. An
 *  agent whose step_dt silently varies is very hard to debug. The guard stops a
 *  world whose clock has stopped from spinning here forever. */
void GymServer::stepTicks(int ticks)
{
    World *world = World::getWorld();
    if (world == NULL) return;
    const int target = world->getTicksSinceStart() + ticks;
    int guard = ticks * 4 + 8;
    while (guard-- > 0)
    {
        world = World::getWorld();
        if (world == NULL || world->getTicksSinceStart() >= target) return;
        main_loop->updateSingleTick(false/*fast_forward*/, NULL);
    }
    Log::warn("GymServer", "The race clock did not advance %d ticks; the race "
                           "may be over.", ticks);
}   // stepTicks

//-----------------------------------------------------------------------------
/** Draws one frame in the windowed mode, so that watching a policy shows a race
 *  rather than a frozen first frame. These are the same calls MainLoop::run
 *  makes, minus input handling: the window is a view, not a controller. */
void GymServer::updateGraphics()
{
#ifndef SERVER_ONLY
    if (GUIEngine::isNoGraphics()) return;
    World *world = World::getWorld();
    if (world == NULL) return;
    const float frame_duration = stk_config->ticks2Time(m_frame_skip);
    world->updateGraphics(frame_duration);
    irr_driver->update(frame_duration);
    if (irr_driver->getDevice())
        irr_driver->getDevice()->run();
#endif
}   // updateGraphics

//-----------------------------------------------------------------------------
std::string GymServer::handle(const std::string &line)
{
    GymJson::Value request;
    std::string parse_error;
    if (!GymJson::parse(line, &request, &parse_error))
    {
        // A client bug must not take the server down: report and carry on.
        return error(0, ERR_BAD_JSON, parse_error);
    }
    if (!request.isObject())
        return error(0, ERR_BAD_JSON, "the request must be a JSON object");

    int id = 0;
    std::string env_error;
    if (!checkEnvId(request, &id, &env_error)) return env_error;

    const std::string cmd = request.get("cmd").asString();

    // -- hello ---------------------------------------------------------------
    if (cmd == "hello")
    {
        GymJson::Writer writer;
        writer.addInt   ("id",       id);
        writer.addBool  ("ok",       true);
        writer.beginObject("meta");
        writer.addInt   ("protocol",     PROTOCOL_VERSION);
        writer.addString("game",         "supertuxkart");
        writer.addString("track",        RaceManager::get()->getTrackName());
        writer.addInt   ("laps",         RaceManager::get()->getNumLaps());
        writer.addInt   ("difficulty",   (int)RaceManager::get()
                                                          ->getDifficulty());
        writer.addInt   ("num_karts",    (int)World::getWorld()->getNumKarts());
        writer.addInt   ("kart_id",      (int)m_kart_id);
        writer.addInt   ("physics_fps",  stk_config->getPhysicsFPS());
        writer.addFloat ("tick_dt",      stk_config->ticks2Time(1));
        writer.addInt   ("frames",       m_frame_skip);
        writer.addFloat ("step_dt",      stk_config->ticks2Time(m_frame_skip));
        writer.addInt   ("lookahead_k",  m_lookahead_k);
        // One less than num_karts: the agent's own kart is not in the array.
        writer.addInt   ("max_karts",    m_include_karts
                                       ? (int)World::getWorld()->getNumKarts()-1
                                       : 0);
        writer.addFloat ("track_length", GymState::getTrackLength());
        writer.addBool  ("render",       !GUIEngine::isNoGraphics());
        writer.addBool  ("expert",       m_expert);
        writer.addBool  ("human",        m_human);
        writer.endObject();
        return writer.toString();
    }

    // -- reset ---------------------------------------------------------------
    if (cmd == "reset" || cmd == "reset_batch")
    {
        if (cmd == "reset_batch")
        {
            const GymJson::Value &env_ids = request.get("env_ids");
            if (!env_ids.isArray() || env_ids.getArray().size() != 1 ||
                env_ids.getArray()[0].asInt(-1) != 0)
            {
                return error(id, ERR_BAD_BATCH,
                             "this build holds one environment per process, so "
                             "a batch must be exactly [0]");
            }
        }
        if (request.has("options") &&
            request.get("options").has("track"))
        {
            return error(id, ERR_NOT_SUPPORTED,
                         "the track is fixed for the life of the process; "
                         "start another child to race a different track");
        }

        const GymJson::Value &seed = request.get("seed");
        if (seed.isNumber())
        {
            const int s = seed.asInt(0);
            RandomGenerator::seed(s);
            srand((unsigned int)s);
        }

        // Restart in place rather than reloading the track, which would cost
        // seconds per episode.
        resetRace();
        m_has_reset = true;

        return cmd == "reset" ? buildState(id) : buildStateBatch(id);
    }

    // -- step ----------------------------------------------------------------
    if (cmd == "step" || cmd == "step_batch")
    {
        if (m_human)
        {
            return error(id, ERR_NOT_SUPPORTED,
                         "in --gym-human mode the race runs on its own clock; "
                         "poll it with \"state\" instead of stepping it");
        }
        if (!m_has_reset)
            return error(id, ERR_NOT_RESET, "reset must be called before step");

        const GymJson::Value *action = NULL;
        if (cmd == "step_batch")
        {
            const GymJson::Value &actions = request.get("actions");
            if (!actions.isArray() || actions.getArray().size() != 1)
            {
                return error(id, ERR_BAD_BATCH,
                             "this build holds one environment per process, so "
                             "a batch must hold exactly one action");
            }
            action = &actions.getArray()[0];
        }
        else
            action = &request.get("action");

        if (!action->isNull() && !action->isObject())
        {
            return error(id, ERR_BAD_ACTION,
                         "action must be an object of control values, e.g. "
                         "{\"steer\":-0.3,\"accel\":1.0}");
        }
        applyAction(*action);
        stepTicks(m_frame_skip);
        updateGraphics();
        return cmd == "step" ? buildState(id) : buildStateBatch(id);
    }

    // -- state ---------------------------------------------------------------
    if (cmd == "state")
    {
        if (!m_has_reset)
            return error(id, ERR_NOT_RESET,
                         "reset must be called before state");
        return buildState(id);
    }

    // -- close ---------------------------------------------------------------
    if (cmd == "close")
    {
        m_quit = true;
        if (m_human && main_loop) main_loop->requestAbort();
        GymJson::Writer writer;
        writer.addInt ("id", id);
        writer.addBool("ok", true);
        return writer.toString();
    }

    if (cmd.empty())
        return error(id, ERR_UNKNOWN_CMD, "the request has no \"cmd\" field");
    return error(id, ERR_UNKNOWN_CMD, "unknown command \"" + cmd + "\"");
}   // handle
