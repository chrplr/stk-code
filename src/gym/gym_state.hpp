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

#ifndef HEADER_GYM_STATE_HPP
#define HEADER_GYM_STATE_HPP

namespace GymJson { class Writer; }

/** Writes what the agent's kart can observe into a protocol response.
 *
 *  Facts only: no reward, no "terminated", no normalisation. The reward scheme,
 *  the termination rule and the observation tensor all live in the Python
 *  client so that changing any of them costs an edit rather than a rebuild of
 *  this binary - the same rule the Rush-Hour and agario servers follow.
 *
 *  This is the only file that reaches into the rest of STK for observations. */
namespace GymState
{
    /** Writes the state's members into whatever object the caller has already
     *  opened. Split from write() so that a batch response can put the same
     *  fields into an unnamed array element without a second copy of the list.
     *  \param writer         The response being built.
     *  \param kart_id        World id of the kart being observed.
     *  \param lookahead_k    Number of driveline points ahead to report.
     *  \param include_karts  Whether to add the per-opponent array. */
    void writeFields(GymJson::Writer *writer, unsigned int kart_id,
                     unsigned int lookahead_k, bool include_karts);

    /** Writes the members above as a "state" member of the response. */
    void write(GymJson::Writer *writer, unsigned int kart_id,
               unsigned int lookahead_k, bool include_karts);

    /** Total length of the driveline, reported once in the handshake. Returns 0
     *  if the track has no drive graph (an arena). */
    float getTrackLength();
}   // namespace GymState

#endif
