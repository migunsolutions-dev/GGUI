import json
import pathlib


def main() -> None:
    data_path = pathlib.Path(r"c:\Users\migun\Desktop\GGUI\_mesh_profile_analysis.json")
    out_path = pathlib.Path(
        r"C:\Users\migun\.cursor\projects\c-Users-migun-Desktop-GGUI\canvases\mesh-evolution-comparison.canvas.tsx"
    )
    data = json.loads(data_path.read_text(encoding="utf-8"))
    js = json.dumps(data, ensure_ascii=False)

    code = f"""import {{ Callout, Grid, H1, H2, Row, Select, Stack, Stat, Text, useCanvasState, useHostTheme }} from "cursor/canvas";

const DATA = {js} as const;

function polylinePoints(values: Array<number | null>, width: number, height: number, yMin: number, yMax: number) {{
  const n = values.length;
  const pts: string[] = [];
  for (let i = 0; i < n; i += 1) {{
    const v = values[i];
    if (v == null) continue;
    const x = (i / Math.max(1, n - 1)) * width;
    const y = height - ((v - yMin) / Math.max(1e-9, yMax - yMin)) * height;
    pts.push(`${{x.toFixed(2)}},${{y.toFixed(2)}}`);
  }}
  return pts.join(" ");
}}

function valueAtDistance(profile: Array<number | null>, distStep: number, d: number) {{
  const idx = Math.max(0, Math.min(profile.length - 1, Math.round(d / distStep)));
  return profile[idx];
}}

function shiftPoints(points: string, sx: number, sy: number) {{
  if (!points) return "";
  return points
    .split(" ")
    .map((p) => {{
      const [x, y] = p.split(",").map(Number);
      return `${{x + sx}},${{y + sy}}`;
    }})
    .join(" ");
}}

export default function MeshEvolutionComparison() {{
  const theme = useHostTheme();
  const commonTimes = DATA.common_times as number[];
  const [timeIdxStr, setTimeIdxStr] = useCanvasState<string>("time-index", "0");
  const [distanceProbe, setDistanceProbe] = useCanvasState<string>("distance-probe", "1.0");

  const maxIdx = Math.max(0, commonTimes.length - 1);
  const timeIdx = Math.max(0, Math.min(maxIdx, Number(timeIdxStr) || 0));
  const t = commonTimes[timeIdx] ?? 0;

  const directSeries = DATA.cases.direct.profiles as Array<any>;
  const guiSeries = DATA.cases.gui.profiles as Array<any>;
  const direct = directSeries.find((p) => p.time === t) ?? directSeries[0];
  const gui = guiSeries.find((p) => p.time === t) ?? guiSeries[0];

  const allVals = [...(direct?.profile ?? []), ...(gui?.profile ?? [])].filter((v) => v != null) as number[];
  const yMin = allVals.length ? Math.min(...allVals) : 0;
  const yMax = allVals.length ? Math.max(...allVals) : 1;

  const width = 860;
  const height = 260;
  const distStep = Number(DATA.meta.sample_step) || 0.05;

  const dProbe = Math.max(0, Math.min(5, Number(distanceProbe) || 1));
  const vDirectProbe = valueAtDistance(direct.profile, distStep, dProbe);
  const vGuiProbe = valueAtDistance(gui.profile, distStep, dProbe);

  const nearBand = 1.0;
  const nearIdx = Math.round(nearBand / distStep);
  const nearDirect = directSeries
    .map((p) => {{
      const vals = p.profile.slice(0, nearIdx + 1).filter((v: number | null) => v != null) as number[];
      return vals.length ? {{ time: p.time, avg: vals.reduce((a, b) => a + b, 0) / vals.length }} : null;
    }})
    .filter(Boolean) as Array<{{ time: number; avg: number }}>;
  const nearGui = guiSeries
    .map((p) => {{
      const vals = p.profile.slice(0, nearIdx + 1).filter((v: number | null) => v != null) as number[];
      return vals.length ? {{ time: p.time, avg: vals.reduce((a, b) => a + b, 0) / vals.length }} : null;
    }})
    .filter(Boolean) as Array<{{ time: number; avg: number }}>;

  const tMin = Math.min(...commonTimes);
  const tMax = Math.max(...commonTimes);
  const nearVals = [...nearDirect.map((x) => x.avg), ...nearGui.map((x) => x.avg)];
  const nMin = nearVals.length ? Math.min(...nearVals) : 0;
  const nMax = nearVals.length ? Math.max(...nearVals) : 1;

  const nearLine = (series: Array<{{ time: number; avg: number }}>, w: number, h: number) =>
    series
      .map((p) => {{
        const x = ((p.time - tMin) / Math.max(1e-12, tMax - tMin)) * w;
        const y = h - ((p.avg - nMin) / Math.max(1e-9, nMax - nMin)) * h;
        return `${{x.toFixed(2)}},${{y.toFixed(2)}}`;
      }})
      .join(" ");

  return (
    <Stack gap={{18}}>
      <H1>Mesh Evolution: Direct vs GUI</H1>
      <Text tone="secondary">Comparison along a radial line from charge center to 5 m (direction chosen to avoid crossing obstacle wall).</Text>

      <Grid columns={{4}} gap={{12}}>
        <Stat value={{DATA.cases.direct.runtime_s?.toFixed(2) ?? "n/a"}} label="Direct runtime [s]" />
        <Stat value={{DATA.cases.gui.runtime_s?.toFixed(2) ?? "n/a"}} label="GUI runtime [s]" />
        <Stat value={{String(DATA.cases.direct.n_times)}} label="Saved time steps" />
        <Stat value={{t.toExponential(3)}} label="Current time" />
      </Grid>

      <Row gap={{10}} align="center">
        <Text weight="semibold">Saved step:</Text>
        <Select
          value={{String(timeIdx)}}
          onChange={{setTimeIdxStr}}
          options={{commonTimes.map((ct, idx) => ({{ value: String(idx), label: `${{idx + 1}} / ${{commonTimes.length}}   t=${{ct.toExponential(3)}}` }}))}}
          style={{{{ minWidth: 280 }}}}
        />
        <Text weight="semibold">Probe distance [m]:</Text>
        <Select
          value={{distanceProbe}}
          onChange={{setDistanceProbe}}
          options={{["0.5", "1.0", "1.5", "2.0", "3.0", "4.0"].map((v) => ({{ value: v, label: v }}))}}
          style={{{{ width: 100 }}}}
        />
      </Row>

      <H2>Element Size vs Distance at Selected Time</H2>
      <svg width="900" height="320" viewBox="0 0 900 320" role="img" aria-label="Element size versus distance">
        <rect x="40" y="20" width="860" height="260" fill={{theme.bg.elevated}} stroke={{theme.stroke.secondary}} />
        <polyline fill="none" stroke={{theme.accent.primary}} strokeWidth="2" points={{shiftPoints(polylinePoints(direct.profile, width, height, yMin, yMax), 40, 20)}} />
        <polyline fill="none" stroke={{theme.text.secondary}} strokeWidth="2" points={{shiftPoints(polylinePoints(gui.profile, width, height, yMin, yMax), 40, 20)}} />
        <text x="46" y="16" fill={{theme.text.secondary}} fontSize="11">cell size [m]</text>
        <text x="785" y="312" fill={{theme.text.secondary}} fontSize="11">distance [m]</text>
      </svg>

      <Row gap={{18}} align="center">
        <Row gap={{6}} align="center"><div style={{{{ width: 12, height: 3, background: theme.accent.primary }}}} /><Text>Direct example</Text></Row>
        <Row gap={{6}} align="center"><div style={{{{ width: 12, height: 3, background: theme.text.secondary }}}} /><Text>GUI-generated from same example</Text></Row>
      </Row>

      <Grid columns={{2}} gap={{12}}>
        <Stat value={{vDirectProbe == null ? "n/a" : vDirectProbe.toFixed(4)}} label={{`Direct size at ${{dProbe.toFixed(1)}} m`}} />
        <Stat value={{vGuiProbe == null ? "n/a" : vGuiProbe.toFixed(4)}} label={{`GUI size at ${{dProbe.toFixed(1)}} m`}} />
      </Grid>

      <H2>Near-source Average Size (0-1 m) Through Time</H2>
      <svg width="900" height="280" viewBox="0 0 900 280" role="img" aria-label="Near-source average size through time">
        <rect x="40" y="20" width="860" height="220" fill={{theme.bg.elevated}} stroke={{theme.stroke.secondary}} />
        <polyline fill="none" stroke={{theme.accent.primary}} strokeWidth="2" points={{shiftPoints(nearLine(nearDirect, 860, 220), 40, 20)}} />
        <polyline fill="none" stroke={{theme.text.secondary}} strokeWidth="2" points={{shiftPoints(nearLine(nearGui, 860, 220), 40, 20)}} />
      </svg>

      <Callout tone="info" title="Interpretation basis">
        Element size is derived from cellLevel as dx = 0.5 / 2^level along the selected line.
        Lower value means finer mesh.
      </Callout>
    </Stack>
  );
}}
"""
    out_path.write_text(code, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
