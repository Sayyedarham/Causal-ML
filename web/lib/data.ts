// Server-side artifact loading. Everything is read from web/data at build time
// (static export), so the deployed site has no runtime data dependency.
import { readFile } from "node:fs/promises";
import path from "node:path";
import type {
  Allocation, CalibrationBin, Curve, Dataset, Effects, FeatureImportance,
  ModelManifest, ModelMeta, Randomization, RunInfo, TopKRow,
} from "./types";

const DATA_DIR = path.join(process.cwd(), "data");

async function load<T>(file: string): Promise<T> {
  return JSON.parse(await readFile(path.join(DATA_DIR, file), "utf8")) as T;
}

export const getDataset = () => load<Dataset>("dataset.json");
export const getRun = () => load<RunInfo>("run.json");
export const getEffects = () => load<Effects>("effects.json");
export const getRandomization = () => load<Randomization>("randomization.json");
export const getCurves = () => load<Record<string, Curve>>("curves.json");
export const getTopK = () => load<TopKRow[]>("topk.json");
export const getCalibration = () => load<Record<string, CalibrationBin[]>>("calibration.json");
export const getModelMeta = () => load<Record<string, ModelMeta>>("models.json");
export const getFeatureImportance = () => load<Record<string, FeatureImportance[]>>("feature_importance.json");
export const getAllocation = () => load<Allocation>("allocation.json");
export const getExposure = () => load<import("./types").Exposure>("exposure.json");
export const getModelManifest = () => load<ModelManifest>("models/manifest.json");

export const CAUSAL_ORDER = ["dr_learner", "x_learner", "t_learner", "s_learner", "constant"];
export const ALL_ORDER = ["dr_learner", "x_learner", "t_learner", "s_learner", "constant", "response"];
