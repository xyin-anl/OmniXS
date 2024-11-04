# %%
import pandas as pd


def create_xas_df(model_data, win_rates):
    rows = []
    for tag, value in model_data.items():
        element_key = f"{tag.element} {tag.type if tag.type == 'VASP' else ''}"

        # Find matching expert tag for win rates
        expert_tag = next(
            (
                t
                for t in win_rates.keys()
                if t.element == tag.element and t.type == tag.type
            ),
            None,
        )

        # Find related values
        related_values = {
            t.name: v
            for t, v in model_data.items()
            if t.element == tag.element and t.type == tag.type
        }

        if tag.name == "expertXAS":
            row = {
                "Element": element_key.strip(),
                "ExpertXAS": value,
                "UniversalXAS": related_values.get("universalXAS", "N/A"),
                "Tuned-UniversalXAS": related_values["tunedUniversalXAS"],
                "Improvement (%)": (related_values["tunedUniversalXAS"] - value)
                / value
                * 100,
                "Energy": win_rates.get(expert_tag, {}).get("energy", None),
                "Spectra": win_rates.get(expert_tag, {}).get("spectra", None),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    for col in df.columns[1:]:  # Skip Element column
        df[col] = df[col].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) and x != "N/A" else x
        )

    return df


if __name__ == "__main__":
    from omnixas.scripts.plots.scripts import AllEtas, ExpertTunedWinRates

    model_data = AllEtas()
    win_rates = ExpertTunedWinRates()

    df = create_xas_df(model_data, win_rates)
    with open("performance_table.md", "w") as f:
        f.write(df.to_markdown(index=False))

# %%
