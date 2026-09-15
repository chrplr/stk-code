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

#ifndef HEADER_GYM_CONTROLLER_HPP
#define HEADER_GYM_CONTROLLER_HPP

#include "input/input.hpp"
#include "karts/controller/kart_control.hpp"
#include "karts/controller/controller.hpp"
#include "karts/controller/player_controller.hpp"
#include "karts/controller/skidding_ai.hpp"

/** The kart control values an external agent asks for.
 *
 *  This is the whole of STK's control surface, deliberately: the server accepts
 *  every field and the Python client decides which subset to expose as a gym
 *  action space, the same way it decides the reward and the observation
 *  encoding. Adding a discrete action set costs a Python edit, not a rebuild. */
struct GymAction
{
    float m_steer;
    float m_accel;
    bool  m_brake;
    bool  m_nitro;
    /** One of KartControl::SkidControl. */
    int   m_skid;
    bool  m_fire;
    bool  m_rescue;

    GymAction() { reset(); }

    void reset()
    {
        m_steer  = 0.0f;
        m_accel  = 0.0f;
        m_brake  = false;
        m_nitro  = false;
        m_skid   = KartControl::SC_NONE;
        m_fire   = false;
        m_rescue = false;
    }   // reset
};   // struct GymAction

/** A controller whose actions come from outside the process.
 *
 *  It holds the last action received over the gym protocol and re-applies it on
 *  every physics tick, so that one agent step covers several ticks of the same
 *  action (the usual frame-skip). It reports itself as a player controller,
 *  because it occupies the player slot: the AI rubber-bands against it and
 *  StandardRace waits for it to finish rather than ending the race early. It is
 *  not a *local* player controller, which keeps sound, achievements and input
 *  devices out of the picture; the camera it needs for the windowed mode is
 *  created here explicitly, the way ProfileWorld does it.
 *
 * \ingroup controller
 */
class GymController : public Controller
{
private:
    /** The action to apply, replaced whenever a step command arrives. */
    GymAction m_action;

public:
             GymController(AbstractKart *kart);
    virtual ~GymController() {}

    /** Replaces the action applied from the next tick onwards. */
    void setAction(const GymAction &action) { m_action = action; }
    // ------------------------------------------------------------------------
    const GymAction& getAction() const      { return m_action;   }
    // ------------------------------------------------------------------------
    virtual void reset() OVERRIDE;
    virtual void update(int ticks) OVERRIDE;
    virtual void finishedRace(float time) OVERRIDE {}
    virtual void crashed(const AbstractKart *k) OVERRIDE {}
    virtual void crashed(const Material *m) OVERRIDE {}
    virtual void handleZipper(bool play_sound) OVERRIDE {}
    virtual void collectedItem(const ItemState &item,
                               float previous_energy = 0) OVERRIDE {}
    virtual void setPosition(int p) OVERRIDE {}
    virtual void newLap(int lap) OVERRIDE {}
    virtual void skidBonusTriggered() OVERRIDE {}
    virtual bool disableSlipstreamBonus() const OVERRIDE { return false; }
    // ------------------------------------------------------------------------
    /** The kart occupies the player slot, so the rest of the game must treat it
     *  as a player: see StandardRace::endRaceEarly. */
    virtual bool isPlayerController() const OVERRIDE      { return true;  }
    // ------------------------------------------------------------------------
    /** But not a local one: no camera-by-default, no sound, no achievements,
     *  no input device. */
    virtual bool isLocalPlayerController() const OVERRIDE { return false; }
    // ------------------------------------------------------------------------
    /** Actions arrive over the protocol, not through the input system. */
    virtual bool action(PlayerAction action, int value,
                        bool dry_run = false) OVERRIDE   { return true;  }
    // ------------------------------------------------------------------------
    virtual bool saveState(BareNetworkString *buffer) const OVERRIDE;
    virtual void rewindTo(BareNetworkString *buffer) OVERRIDE;
};   // class GymController

/** The player's seat driven by named keys rather than by control values.
 *
 *  A keyboard does not set a steering angle: it sends press and release
 *  events, and PlayerController turns those into the ramped steering, the
 *  latched skid direction and the nitro-only-while-accelerating that a person
 *  playing SuperTuxKart feels. This controller is that PlayerController, with
 *  the events arriving over the protocol: a step carries the set of keys held
 *  and every change against the previous set becomes one action() call. An
 *  agent trained through it and a person playing through fmri-gym's keyboard
 *  are then subject to the same input rules, which is the point.
 *
 *  Like GymController it is a player but not a local one, and makes its own
 *  camera.
 *
 * \ingroup controller
 */
class GymKeyController : public PlayerController
{
public:
    enum { NUM_KEYS = 8 };
    /** The wire names of the keys, in the order of KEY_ACTIONS. */
    static const char        *KEY_NAMES[NUM_KEYS];
    static const PlayerAction KEY_ACTIONS[NUM_KEYS];

private:
    bool m_held[NUM_KEYS];

public:
             GymKeyController(AbstractKart *kart);
    virtual ~GymKeyController() {}

    /** Applies a new set of held keys: one press or release per change. */
    void setKeys(const bool held[NUM_KEYS]);
    // ------------------------------------------------------------------------
    virtual void reset() OVERRIDE;
    // ------------------------------------------------------------------------
    /** A player, so the AI rubber-bands against it and the race waits for it;
     *  not a local one: no sound, no achievements, no input device. */
    virtual bool isLocalPlayerController() const OVERRIDE { return false; }
    // ------------------------------------------------------------------------
    virtual void finishedRace(float time) OVERRIDE {}
    virtual void crashed(const AbstractKart *k) OVERRIDE {}
    virtual void crashed(const Material *m) OVERRIDE {}
    virtual void setPosition(int p) OVERRIDE {}
    virtual void newLap(int lap) OVERRIDE {}
    virtual bool disableSlipstreamBonus() const OVERRIDE { return false; }
};   // class GymKeyController

/** STK's own racing AI, sitting in the player's seat as a reference policy.
 *
 *  The only thing it changes is that it answers yes to isPlayerController, and
 *  that is not cosmetic: World::getPlayerKart finds the player karts by asking
 *  their controllers, and RaceManager still counts one player, so a plain
 *  SkiddingAI here makes getPlayerKart(0) return NULL and the other AIs
 *  dereference it in computeNearestKarts. Saying yes also means the other karts
 *  rubber-band against this one exactly as they do against a learning agent,
 *  which is what makes the two comparable.
 *
 * \ingroup controller
 */
class GymExpertController : public SkiddingAI
{
public:
    GymExpertController(AbstractKart *kart) : SkiddingAI(kart)
    {
        setControllerName("GymExpertController");
    }
    // ------------------------------------------------------------------------
    virtual bool isPlayerController() const OVERRIDE { return true; }
};   // class GymExpertController

#endif
