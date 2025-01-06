
from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .forms import FileUploadForm
import os
from django.shortcuts import render
import scipy.stats as stats
import numpy as np
from django.shortcuts import render
from scipy import stats
import numpy as np
from django.shortcuts import render
from scipy import stats
import numpy as np
from django.shortcuts import render
from scipy import stats
import numpy as np

import io
import base64
from scipy.stats import binom, bernoulli, poisson, uniform, expon
from django.shortcuts import render

def accueil(request):
    return render(request, 'stats/accueil.html')

def handle_file(file):
    file_extension = os.path.splitext(file.name)[-1].lower()
    try:
        if file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file, engine='openpyxl' if file_extension == '.xlsx' else 'xlrd')
        elif file_extension == '.csv':
            return pd.read_csv(file)
        else:
            raise ValueError("Format de fichier non supporté.")
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier : {str(e)}")


def upload_file(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            action = request.POST.get('action')  # Récupérer l'action sélectionnée
            try:
                # Traiter le fichier
                data = handle_file(file)
                numeric_columns = data.select_dtypes(include='number').columns.tolist()
                if not numeric_columns:
                    raise ValueError("Le fichier ne contient pas de colonnes numériques.")

                # Stocker les données dans la session
                request.session['data'] = data.to_dict(orient='list')
                request.session['numeric_columns'] = numeric_columns

                # Rediriger en fonction de l'action choisie
                if action == 'visualize':
                    return redirect('visualize')
                elif action == 'statistics':
                    return redirect('perform_statistics')

                # Si aucune action n'est spécifiée, afficher un aperçu
                return render(request, 'stats/upload.html', {
                    'form': form,
                    'columns': numeric_columns,
                    'file_content': data.head(10).values.tolist()  # Aperçu des 10 premières lignes
                })
            except ValueError as e:
                return render(request, 'stats/upload.html', {'form': form, 'error': str(e)})

    else:
        form = FileUploadForm()

    return render(request, 'stats/upload.html', {'form': form})


from django.shortcuts import render, redirect
import pandas as pd

def perform_statistics(request):
    # Dictionnaire de mappage des calculs
    CALCULATION_NAMES = {
        'mean': 'Moyenne',
        'median': 'Médiane',
        'variance': 'Variance',
        'std_dev': 'Écart-type',
        'range': 'Étendue',
        'mode': 'Mode',
        'expectation': 'Espérance'
    }

    # Charger les données depuis la session
    data_dict = request.session.get('data')
    numeric_columns = request.session.get('numeric_columns')

    if not data_dict or not numeric_columns:
        return redirect('upload_file')

    data = pd.DataFrame(data_dict)

    if request.method == 'POST':
        # Récupérer les colonnes sélectionnées et les calculs
        selected_columns = request.POST.getlist('columns')
        calculations = request.POST.getlist('calculations')

        if not selected_columns:
            return render(request, 'stats/perform_statistics.html', {
                'columns': numeric_columns,
                'error': "Veuillez sélectionner au moins une colonne."
            })

        results = {}
        try:
            for column in selected_columns:
                col_data = data[column]
                col_results = {}

                # Effectuer les calculs sélectionnés
                for calc in calculations:
                    if calc == 'mean':
                        col_results[CALCULATION_NAMES[calc]] = col_data.mean()
                    elif calc == 'median':
                        col_results[CALCULATION_NAMES[calc]] = col_data.median()
                    elif calc == 'variance':
                        col_results[CALCULATION_NAMES[calc]] = col_data.var()
                    elif calc == 'std_dev':
                        col_results[CALCULATION_NAMES[calc]] = col_data.std()
                    elif calc == 'range':
                        col_results[CALCULATION_NAMES[calc]] = col_data.max() - col_data.min()
                    elif calc == 'mode':
                        mode_result = col_data.mode()
                        col_results[CALCULATION_NAMES[calc]] = mode_result.iloc[0] if not mode_result.empty else "Pas de mode"
                    elif calc == 'expectation':
                        col_results[CALCULATION_NAMES[calc]] = col_data.mean()

                results[column] = col_results

        except Exception as e:
            return render(request, 'stats/perform_statistics.html', {
                'columns': numeric_columns,
                'error': f"Erreur : {str(e)}"
            })

        return render(request, 'stats/perform_statistics.html', {
            'columns': numeric_columns,
            'results': results
        })

    return render(request, 'stats/perform_statistics.html', {'columns': numeric_columns})

from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import binom, bernoulli, poisson, uniform, expon
def perform_laws(request):

    if request.method == 'POST':
        law = request.POST.get('law')
        equation = ""
        substituted_equation = ""
        result_value = None
        image_base64 = None

        try:
            if law == 'binomial':
                n = int(request.POST.get('n', 10))
                p = float(request.POST.get('p', 0.5))
                k = 3  # Fixed single value
                equation = r"P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}"
                substituted_equation = rf"P(X = {k}) = \binom{{{n}}}{{{k}}} {p}^{k} (1 - {p})^{{{n - k}}}"
                result_value = binom.pmf(k, n, p)
                substituted_equation += rf" = {result_value:.5f}"

                x = range(0, n + 1)
                y = binom.pmf(x, n, p)
                fig, ax = plt.subplots()
                ax.bar(x, y, alpha=0.7, color="blue")
                ax.set_title("Loi Binomiale")
                ax.set_xlabel("Valeurs")
                ax.set_ylabel("Probabilité")

            elif law == 'bernoulli':
                p = float(request.POST.get('p', 0.5))
                x_value = 1  # Fixed single value
                equation = r"P(X = x) = p^x (1 - p)^{1 - x}"
                substituted_equation = rf"P(X = {x_value}) = {p}^{x_value} (1 - {p})^{1 - x_value}"
                result_value = bernoulli.pmf(x_value, p)
                substituted_equation += rf" = {result_value:.5f}"

                x = [0, 1]
                y = bernoulli.pmf(x, p)
                fig, ax = plt.subplots()
                ax.bar(x, y, alpha=0.7, color="green")
                ax.set_title("Loi de Bernoulli")
                ax.set_xlabel("Valeurs")
                ax.set_ylabel("Probabilité")

            elif law == 'poisson':
                lambda_ = float(request.POST.get('lambda', 5))
                k = 3  # Fixed single value
                equation = r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}"
                substituted_equation = rf"P(X = {k}) = \frac{{{lambda_}^{k} e^{{-{lambda_}}}}}{{{k}!}}"
                result_value = poisson.pmf(k, lambda_)
                substituted_equation += rf" = {result_value:.5f}"

                x = range(0, 20)
                y = poisson.pmf(x, lambda_)
                fig, ax = plt.subplots()
                ax.bar(x, y, alpha=0.7, color="orange")
                ax.set_title("Loi de Poisson")
                ax.set_xlabel("Valeurs")
                ax.set_ylabel("Probabilité")

            elif law == 'uniform_discrete':
                a = int(request.POST.get('a', 0))
                b = int(request.POST.get('b', 10))
                k = 5  # Fixed single value
                equation = r"P(X = x) = \frac{1}{b - a + 1}"
                substituted_equation = rf"P(X = {k}) = \frac{{1}}{{{b} - {a} + 1}}"
                result_value = 1 / (b - a + 1)
                substituted_equation += rf" = {result_value:.5f}"

                x = range(a, b + 1)
                y = [1 / (b - a + 1)] * len(x)
                fig, ax = plt.subplots()
                ax.bar(x, y, alpha=0.7, color="purple")
                ax.set_title("Loi Uniforme Discrète")
                ax.set_xlabel("Valeurs")
                ax.set_ylabel("Probabilité")

            elif law == 'uniform_continuous':
                a = float(request.POST.get('a', 0))
                b = float(request.POST.get('b', 1))
                x_value = 0.5  # Fixed single value
                equation = r"f(x) = \frac{1}{b - a}"
                substituted_equation = rf"f({x_value}) = \frac{{1}}{{{b} - {a}}}"
                result_value = uniform.pdf(x_value, loc=a, scale=b - a)
                substituted_equation += rf" = {result_value:.5f}"

                x = np.linspace(a, b, 100)
                y = uniform.pdf(x, loc=a, scale=b - a)
                fig, ax = plt.subplots()
                ax.plot(x, y, color="cyan")
                ax.set_title("Loi Uniforme Continue")
                ax.set_xlabel("Valeurs")
                ax.set_ylabel("Densité")

            elif law == 'exponential':
                scale = float(request.POST.get('scale', 1))
                x_value = 1  # Fixed single value
                equation = r"f(x) = \lambda e^{-\lambda x}"
                substituted_equation = rf"f({x_value}) = (1/{scale}) e^{{-(1/{scale}) * {x_value}}}"
                result_value = expon.pdf(x_value, scale=scale)
                substituted_equation += rf" = {result_value:.5f}"

                x = np.linspace(0, 5 * scale, 100)
                y = expon.pdf(x, scale=scale)
                fig, ax = plt.subplots()
                ax.plot(x, y, color="red")
                ax.set_title("Loi Exponentielle")
                ax.set_xlabel("Valeurs")
                ax.set_ylabel("Densité")

            elif law == 'normal':
                mu = float(request.POST.get('mu', 0))
                sigma = float(request.POST.get('sigma', 1))
                x_value = mu  # Fixed single value for simplicity
                equation = r"f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}"
                substituted_equation = rf"f({x_value}) = \frac{{1}}{{\sqrt{{2\pi{sigma}^2}}}} e^{{-\frac{{({x_value} - {mu})^2}}{{2*{sigma}^2}}}}"
                result_value = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x_value - mu) ** 2) / (2 * sigma ** 2))
                substituted_equation += rf" = {result_value:.5f}"

                x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
                y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
                fig, ax = plt.subplots()
                ax.plot(x, y, color="brown")
                ax.set_title("Loi Normale")
                ax.set_xlabel("Valeurs")
                ax.set_ylabel("Densité")

            # Convertir le graphique en image Base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(fig)

            return render(request, 'stats/perform_laws.html', {
                'law': law,
                'equation': equation,
                'substituted_equation': substituted_equation,
                'result_value': f"Result = {result_value:.5f}",
                'image_base64': image_base64
            })

        except Exception as e:
            return render(request, 'stats/perform_laws.html', {
                'error': f"Erreur lors du calcul : {str(e)}"
            })

    return render(request, 'stats/perform_laws.html')

def perform_tests(request):
    result = None
    error = None
    test_type = None
    form_data = {}

    if request.method == 'POST':
        if 'update' in request.POST:  # Bouton Mettre à jour
            form_data = {}
        else:
            try:
                # Étape 1 : Récupération des données
                n = int(request.POST.get('n', ''))
                mean = float(request.POST.get('mean', ''))
                std_dev = float(request.POST.get('std_dev', ''))
                sample_mean = float(request.POST.get('sample_mean', ''))
                confidence = float(request.POST.get('confidence', '95')) / 100

                form_data = {
                    'n': n,
                    'mean': mean,
                    'std_dev': std_dev,
                    'sample_mean': sample_mean,
                    'confidence': confidence * 100
                }

                if n < 2:
                    raise ValueError("La taille de l'échantillon (n) doit être au moins 2.")
                
                # Étape 2 : Sélection du test
                test_type = 'z-test' if n >= 30 else 't-test'

                # Étape 3 : Calcul
                if test_type == 'z-test':
                    z_stat = (sample_mean - mean) / (std_dev / np.sqrt(n))
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    margin_error = stats.norm.ppf(1 - (1 - confidence) / 2) * (std_dev / np.sqrt(n))
                    ci_lower = sample_mean - margin_error
                    ci_upper = sample_mean + margin_error

                    result = {
                        'test': 'Z-Test',
                        'stat': round(z_stat, 3),
                        'p_value': round(p_value, 5),
                        'ci': (round(ci_lower, 3), round(ci_upper, 3)),
                        'equation': "Z = (X̄ - μ) / (σ / √n)",
                        'substituted_equation': f"Z = ({sample_mean} - {mean}) / ({std_dev} / √{n})"
                    }

                elif test_type == 't-test':
                    t_stat = (sample_mean - mean) / (std_dev / np.sqrt(n))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
                    margin_error = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1) * (std_dev / np.sqrt(n))
                    ci_lower = sample_mean - margin_error
                    ci_upper = sample_mean + margin_error

                    result = {
                        'test': 'T-Test',
                        'stat': round(t_stat, 3),
                        'p_value': round(p_value, 5),
                        'ci': (round(ci_lower, 3), round(ci_upper, 3)),
                        'equation': "T = (X̄ - μ) / (σ / √n)",
                        'substituted_equation': f"T = ({sample_mean} - {mean}) / ({std_dev} / √{n})"
                    }

            except Exception as e:
                error = str(e)

    return render(request, 'stats/perform_tests.html', {
        'result': result,
        'error': error,
        'form_data': form_data,
    })

def visualize(request):
    # Charger les données de la session
    data_dict = request.session.get('data')
    numeric_columns = request.session.get('numeric_columns')

    if not data_dict or not numeric_columns:
        return redirect('upload_file')

    data = pd.DataFrame(data_dict)

    if request.method == 'POST':
        selected_columns = request.POST.getlist('columns')
        plot_type = request.POST.get('plot_type')

        if not selected_columns:
            return render(request, 'stats/visualize.html', {
                'columns': numeric_columns,
                'error': "Veuillez sélectionner au moins une colonne pour la visualisation."
            })

        # Générer les graphiques
        fig, ax = plt.subplots()
        try:
            for column in selected_columns:
                if plot_type == 'histogram':
                    sns.histplot(data[column], kde=False, ax=ax, label=column)
                elif plot_type == 'kde':
                    sns.kdeplot(data[column], ax=ax, label=column)
                elif plot_type == 'boxplot':
                    sns.boxplot(x=data[column], ax=ax)
                elif plot_type == 'bar':
                    data[column].value_counts().plot(kind='bar', ax=ax, label=column)
                elif plot_type == 'line':
                    sns.lineplot(data=data, x=data.index, y=column, ax=ax, label=column)

            ax.legend()
            plt.title(f"Visualisation : {plot_type}")

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(fig)

            return render(request, 'stats/visualize.html', {
                'columns': numeric_columns,
                'image_base64': image_base64,
                'selected_columns': selected_columns,
                'plot_type': plot_type
            })

        except Exception as e:
            plt.close(fig)
            return render(request, 'stats/visualize.html', {
                'columns': numeric_columns,
                'error': f"Erreur lors de la visualisation : {str(e)}"
            })

    return render(request, 'stats/visualize.html', {'columns': numeric_columns})
