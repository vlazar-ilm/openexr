//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//
// Once all the pixel data of a part has been written, OutputFile and
// TiledOutputFile release their line / tile buffers (and compressors).
// Check that files written this way are still correct, that extra
// writes are still rejected, and that copyPixels() still works.
//

#include "ImfArray.h"
#include "ImfChannelList.h"
#include "ImfCompression.h"
#include "ImfFrameBuffer.h"
#include "ImfHeader.h"
#include "ImfInputFile.h"
#include "ImfInputPart.h"
#include "ImfMultiPartInputFile.h"
#include "ImfMultiPartOutputFile.h"
#include "ImfOutputFile.h"
#include "ImfOutputPart.h"
#include "ImfPartType.h"
#include "ImfTiledInputPart.h"
#include "ImfTiledOutputFile.h"
#include "ImfTiledOutputPart.h"
#include "ImfThreading.h"
#include "ImathBox.h"
#include "half.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#include "testReleaseBuffers.h"

namespace IMF = OPENEXR_IMF_NAMESPACE;
using namespace IMF;
using namespace IMATH_NAMESPACE;
using namespace std;

namespace
{

const int W = 133;
const int H = 97;

#define CHECK(cond)                                                            \
    do                                                                         \
    {                                                                          \
        if (!(cond))                                                           \
        {                                                                      \
            cerr << "check failed: " #cond " (line " << __LINE__ << ")"        \
                 << endl;                                                      \
            throw std::runtime_error ("testReleaseBuffers failed");            \
        }                                                                      \
    } while (0)

half
pixel (int part, int x, int y, int c)
{
    return half (float ((x * 3 + y * 7 + c * 11 + part * 13) % 64) / 64.f);
}

void
fillPart (vector<half>& buf, int part)
{
    buf.resize (size_t (W) * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c)
                buf[(size_t (y) * W + x) * 3 + c] = pixel (part, x, y, c);
}

FrameBuffer
makeFrameBuffer (vector<half>& buf)
{
    FrameBuffer fb;
    char*       base = (char*) buf.data ();
    const char* names[3] = {"R", "G", "B"};
    for (int c = 0; c < 3; ++c)
        fb.insert (
            names[c],
            Slice (HALF, base + 2 * c, 6, size_t (W) * 6));
    return fb;
}

Header
makeHeader (bool tiled, Compression comp, const char* name)
{
    Header h (W, H);
    h.compression () = comp;
    h.setName (name);
    h.setType (tiled ? TILEDIMAGE : SCANLINEIMAGE);
    if (tiled) h.setTileDescription (TileDescription (32, 32, ONE_LEVEL));
    h.channels ().insert ("R", Channel (HALF));
    h.channels ().insert ("G", Channel (HALF));
    h.channels ().insert ("B", Channel (HALF));
    return h;
}

template <class F>
bool
throwsExc (F f)
{
    try
    {
        f ();
    }
    catch (const IEX_NAMESPACE::BaseExc&)
    {
        return true;
    }
    return false;
}

void
verify (const string& fn, int numParts, Compression comp)
{
    MultiPartInputFile in (fn.c_str ());
    CHECK (in.parts () == numParts);

    for (int p = 0; p < numParts; ++p)
    {
        vector<half> got (size_t (W) * H * 3);
        FrameBuffer  fb = makeFrameBuffer (got);

        if (in.header (p).type () == TILEDIMAGE)
        {
            TiledInputPart ip (in, p);
            ip.setFrameBuffer (fb);
            ip.readTiles (0, ip.numXTiles () - 1, 0, ip.numYTiles () - 1);
        }
        else
        {
            InputPart ip (in, p);
            ip.setFrameBuffer (fb);
            ip.readPixels (0, H - 1);
        }

        //
        // DWA is lossy, so only compare exactly for lossless codecs.
        //

        if (comp != DWAA_COMPRESSION && comp != DWAB_COMPRESSION)
        {
            vector<half> want;
            fillPart (want, p);
            for (size_t i = 0; i < want.size (); ++i)
                CHECK (got[i].bits () == want[i].bits ());
        }
    }
}

void
writeAndCheck (const string& fn, Compression comp, int threads)
{
    setGlobalThreadCount (threads);

    const int         numParts = 4;
    vector<Header>    headers;
    vector<vector<half>> bufs (numParts);

    for (int p = 0; p < numParts; ++p)
    {
        char name[16];
        snprintf (name, sizeof (name), "part%d", p);
        headers.push_back (makeHeader (p % 2 == 1, comp, name));
        fillPart (bufs[p], p);
    }

    {
        MultiPartOutputFile out (fn.c_str (), headers.data (), numParts, false, threads);

        for (int p = 0; p < numParts; ++p)
        {
            FrameBuffer fb = makeFrameBuffer (bufs[p]);

            if (p % 2 == 1)
            {
                TiledOutputPart op (out, p);
                op.setFrameBuffer (fb);
                op.writeTiles (0, op.numXTiles () - 1, 0, op.numYTiles () - 1);

                // all tiles written: more writes must still be rejected
                CHECK (throwsExc ([&] {
                    op.writeTiles (0, 0, 0, 0);
                }));
            }
            else
            {
                OutputPart op (out, p);
                op.setFrameBuffer (fb);
                op.writePixels (H);

                // all scan lines written: more writes must still be rejected
                CHECK (throwsExc ([&] { op.writePixels (1); }));
            }
        }
    }

    verify (fn, numParts, comp);
}

//
// Partial writes must not release anything: write in several calls.
//

void
partialWrites (const string& fn, Compression comp, int threads)
{
    setGlobalThreadCount (threads);

    vector<half> buf;
    fillPart (buf, 0);
    FrameBuffer fb = makeFrameBuffer (buf);

    {
        Header     h = makeHeader (false, comp, "p0");
        OutputFile out (fn.c_str (), h, threads);
        out.setFrameBuffer (fb);
        for (int y = 0; y < H; y += 5)
            out.writePixels (std::min (5, H - y));
        CHECK (throwsExc ([&] { out.writePixels (1); }));
    }

    {
        Header          h = makeHeader (true, comp, "p0");
        TiledOutputFile out (fn.c_str (), h, threads);
        out.setFrameBuffer (fb);
        for (int ty = 0; ty < out.numYTiles (); ++ty)
            out.writeTiles (0, out.numXTiles () - 1, ty, ty);
        CHECK (throwsExc ([&] { out.writeTiles (0, 0, 0, 0); }));
    }
}

//
// copyPixels() on a scanline file and on a tiled file.
//

void
copyChecks (const string& src, const string& dst, Compression comp, int threads)
{
    setGlobalThreadCount (threads);

    vector<half> buf;
    fillPart (buf, 0);
    FrameBuffer fb = makeFrameBuffer (buf);

    {
        Header     h = makeHeader (false, comp, "p0");
        OutputFile out (src.c_str (), h, threads);
        out.setFrameBuffer (fb);
        out.writePixels (H);
    }

    {
        InputFile  in (src.c_str ());
        Header     h = in.header ();
        OutputFile out (dst.c_str (), h, threads);
        out.copyPixels (in);
        CHECK (throwsExc ([&] { out.writePixels (1); }));
    }

    InputFile chk (dst.c_str ());
    vector<half> got (size_t (W) * H * 3);
    FrameBuffer  gfb = makeFrameBuffer (got);
    chk.setFrameBuffer (gfb);
    chk.readPixels (0, H - 1);
    if (comp != DWAA_COMPRESSION && comp != DWAB_COMPRESSION)
        for (size_t i = 0; i < buf.size (); ++i)
            CHECK (got[i].bits () == buf[i].bits ());
}

} // namespace

void
testReleaseBuffers (const string& tempDir)
{
    cout << "Testing that line / tile buffers are released after "
            "all pixel data is written"
         << endl;

    string fn  = tempDir + "imf_test_release_buffers.exr";
    string fn2 = tempDir + "imf_test_release_buffers_copy.exr";

    try
    {
        const Compression comps[] = {
            ZIP_COMPRESSION, PIZ_COMPRESSION, DWAA_COMPRESSION, DWAB_COMPRESSION};
        const int threads[] = {0, 1, 4};

        for (Compression c : comps)
            for (int t : threads)
            {
                cout << "  comp " << int (c) << ", threads " << t << endl;
                writeAndCheck (fn, c, t);
                partialWrites (fn, c, t);
                copyChecks (fn, fn2, c, t);
            }
    }
    catch (...)
    {
        remove (fn.c_str ());
        remove (fn2.c_str ());
        throw;
    }

    remove (fn.c_str ());
    remove (fn2.c_str ());

    cout << "ok\n" << endl;
}
