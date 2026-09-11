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

#include "gym/gym_state.hpp"

#include "gym/gym_json.hpp"
#include "items/powerup.hpp"
#include "karts/abstract_kart.hpp"
#include "karts/controller/kart_control.hpp"
#include "karts/controller/controller.hpp"
#include "modes/linear_world.hpp"
#include "modes/world.hpp"
#include "race/race_manager.hpp"
#include "tracks/drive_graph.hpp"
#include "tracks/graph.hpp"
#include "tracks/drive_node.hpp"
#include "tracks/track_sector.hpp"
#include "utils/vec3.hpp"

#include <string>

#include <vector>

namespace GymState
{

namespace
{
    //-------------------------------------------------------------------------
    /** Converts a world point into the kart's frame: x to the kart's right, y
     *  up, z forward. Everything an agent needs to steer is a relative
     *  quantity, and doing the transform here means the Python side never has
     *  to know STK's world axes. */
    Vec3 toKartLocal(const AbstractKart *kart, const Vec3 &world_point)
    {
        return Vec3(kart->getTrans().inverse() * (btVector3)world_point);
    }   // toKartLocal

    //-------------------------------------------------------------------------
    void addVec3(GymJson::Writer *writer, const std::string &key, const Vec3 &v)
    {
        std::vector<double> values;
        values.push_back(v.getX());
        values.push_back(v.getY());
        values.push_back(v.getZ());
        writer->addFloatArray(key, values);
    }   // addVec3

    //-------------------------------------------------------------------------
    /** Follows the driveline forward from \p start_node, returning the centre
     *  of each of the next \p count nodes. When a node forks, the successor the
     *  AI would take is used, so the lookahead describes the same racing line
     *  SkiddingAI drives. A short track can wrap around, which is correct: the
     *  driveline is a loop. */
    void collectLookahead(int start_node, unsigned int count,
                          std::vector<Vec3> *out)
    {
        out->clear();
        const DriveGraph *graph = DriveGraph::get();
        if (graph == NULL || start_node == Graph::UNKNOWN_SECTOR) return;

        int node = start_node;
        for (unsigned int i = 0; i < count; i++)
        {
            std::vector<unsigned int> successors;
            graph->getSuccessors(node, successors, true/*for_ai*/);
            if (successors.empty()) break;
            node = (int)successors[0];
            out->push_back(graph->getNode(node)->getCenter());
        }
    }   // collectLookahead
}   // anonymous namespace

//-----------------------------------------------------------------------------
float getTrackLength()
{
    const DriveGraph *graph = DriveGraph::get();
    return graph ? graph->getLapLength() : 0.0f;
}   // getTrackLength

//-----------------------------------------------------------------------------
void write(GymJson::Writer *writer, unsigned int kart_id,
           unsigned int lookahead_k, bool include_karts)
{
    writer->beginObject("state");
    writeFields(writer, kart_id, lookahead_k, include_karts);
    writer->endObject();
}   // write

//-----------------------------------------------------------------------------
void writeFields(GymJson::Writer *writer, unsigned int kart_id,
                 unsigned int lookahead_k, bool include_karts)
{
    World *world = World::getWorld();
    if (world == NULL || kart_id >= world->getNumKarts())
    {
        // Should not happen, but a half-written state is worse than an empty
        // one: the client can see the absence of "tick" and report it.
        return;
    }

    AbstractKart *kart = world->getKart(kart_id);
    LinearWorld *linear = dynamic_cast<LinearWorld*>(world);

    // --- race clock ---------------------------------------------------------
    writer->addInt  ("env_id", 0);
    writer->addInt  ("tick",   world->getTicksSinceStart());
    writer->addFloat("time",   world->getTime());
    writer->addInt  ("phase",  (int)world->getPhase());

    // --- the kart itself ----------------------------------------------------
    const KartControl *controls = kart->getController()->getControls();
    writer->addFloat("speed",         kart->getSpeed());
    writer->addFloat("max_speed",     kart->getCurrentMaxSpeed());
    writer->addFloat("steer",         controls ? controls->getSteer() : 0.0f);
    writer->addInt  ("skid",          controls ? (int)controls->getSkidControl()
                                               : 0);
    writer->addFloat("nitro_energy",  kart->getEnergy());
    // What the controller is actually asking of the kart this tick. In agent
    // mode that echoes the last action; with a person at the wheel
    // (--gym-human) it is the only record of what they did.
    writer->beginObject("controls");
    writer->addFloat("steer",     controls ? controls->getSteer()     : 0.0f);
    writer->addFloat("accel",     controls ? controls->getAccel()     : 0.0f);
    writer->addBool ("brake",     controls ? controls->getBrake()     : false);
    writer->addBool ("nitro",     controls ? controls->getNitro()     : false);
    writer->addInt  ("skid",      controls ? (int)controls->getSkidControl()
                                           : 0);
    writer->addBool ("fire",      controls ? controls->getFire()      : false);
    writer->addBool ("rescue",    controls ? controls->getRescue()    : false);
    writer->addBool ("look_back", controls ? controls->getLookBack()  : false);
    writer->endObject();
    writer->addBool ("on_ground",     kart->isOnGround());
    writer->addFloat("heading",       kart->getHeading());
    writer->addFloat("pitch",         kart->getPitch());
    writer->addFloat("roll",          kart->getRoll());
    addVec3(writer, "xyz",            kart->getXYZ());
    addVec3(writer, "velocity_lc",    Vec3(kart->getVelocityLC()));
    addVec3(writer, "normal",         kart->getNormal());
    writer->addInt  ("powerup",       (int)kart->getPowerup()->getType());
    writer->addInt  ("num_powerup",   kart->getNumPowerup());
    writer->addBool ("eliminated",    kart->isEliminated());
    writer->addBool ("finished",      kart->hasFinishedRace());
    writer->addFloat("finish_time",   kart->getFinishTime());
    writer->addInt  ("rank",          kart->getPosition());

    // --- progress along the track -------------------------------------------
    // Only linear race modes have a driveline; an arena has none, and the
    // client is told so by these fields being absent rather than zero.
    if (linear != NULL)
    {
        const TrackSector *sector = linear->getTrackSector(kart_id);
        const int node = sector->getCurrentGraphNode() != Graph::UNKNOWN_SECTOR
                       ? sector->getCurrentGraphNode()
                       : sector->getLastValidGraphNode();

        writer->addBool ("on_road",  sector->isOnRoad());
        // Two different distances, and picking the wrong one is a trap.
        // getDistanceFromStart(false) is recomputed from the kart's position on
        // every tick and is what a progress reward wants. The checkline
        // validated one only moves when a checkline is crossed - it exists to
        // defeat shortcuts - so it is flat between checklines and still holds
        // the previous episode's value at the start of a new one. Both are
        // reported; the client is told which is which by the name.
        writer->addFloat("distance_down_track",
                         sector->getDistanceFromStart(false));
        writer->addFloat("distance_down_track_checked",
                         sector->getDistanceFromStart(true));
        writer->addFloat("distance_to_center",
                         linear->getDistanceToCenterForKart(kart_id));
        // STK's own progress measure, built on the checkline validated
        // distance and therefore inheriting its stepwise behaviour.
        writer->addFloat("overall_distance",
                         linear->getOverallDistance(kart_id));
        writer->addInt  ("finished_laps",
                         linear->getFinishedLapsOfKart(kart_id));
        writer->addInt  ("total_laps",
                         RaceManager::get()->getNumLaps());

        std::vector<Vec3> lookahead;
        collectLookahead(node, lookahead_k, &lookahead);

        // Wrong way is derived here rather than read from LinearWorld, whose
        // timer is private: the kart is going the wrong way when the next
        // driveline point is behind it.
        bool wrong_way = false;
        writer->beginArray("lookahead");
        for (unsigned int i = 0; i < lookahead.size(); i++)
        {
            const Vec3 local = toKartLocal(kart, lookahead[i]);
            if (i == 0) wrong_way = local.getZ() < 0.0f;
            writer->beginArrayElement();
            writer->addFloat("x", local.getX());
            writer->addFloat("y", local.getY());
            writer->addFloat("z", local.getZ());
            writer->endArrayElement();
        }
        writer->endArray();
        writer->addBool("wrong_way", wrong_way);
    }

    // --- the other karts ----------------------------------------------------
    if (include_karts)
    {
        writer->beginArray("karts");
        for (unsigned int i = 0; i < world->getNumKarts(); i++)
        {
            if (i == kart_id) continue;
            AbstractKart *other = world->getKart(i);
            const Vec3 local = toKartLocal(kart, other->getXYZ());
            writer->beginArrayElement();
            writer->addInt  ("id",    (int)i);
            writer->addFloat("x",     local.getX());
            writer->addFloat("y",     local.getY());
            writer->addFloat("z",     local.getZ());
            writer->addFloat("speed", other->getSpeed());
            writer->addInt  ("rank",  other->getPosition());
            writer->addBool ("eliminated", other->isEliminated());
            if (linear != NULL)
            {
                writer->addFloat("distance_down_track",
                    linear->getTrackSector(i)->getDistanceFromStart(false));
                writer->addFloat("overall_distance",
                                 linear->getOverallDistance(i));
                writer->addInt  ("finished_laps",
                                 linear->getFinishedLapsOfKart(i));
            }
            writer->endArrayElement();
        }
        writer->endArray();
    }
}   // writeFields

}   // namespace GymState
