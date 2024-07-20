from scipy.stats import norm, t, chi2
import math

class ConfidenceInterval:
    @staticmethod
    def mean_known(sample_mean: float, sample_size: int, standard_deviation: float, confidence_interval: float) -> tuple[float, float]:
        """Intervalo de confiança para estimar a média, quando a média populacional é conhecida.
        Se n>30, pode utilizar s amostral como desvio padrão populacional"""
        z = float(norm.interval(confidence_interval)[1])
        return (
            sample_mean - z * standard_deviation / math.sqrt(sample_size),
            sample_mean + z * standard_deviation / math.sqrt(sample_size)
        )

    @staticmethod
    def mean_unknown(sample_mean: float, sample_size: int, sample_standard_deviation: float, confidence_interval: float) -> tuple[float, float]:
        """Intervalo de confiança para estimar a média, quando a média populacional é desconhecida"""
        t_positive = float(t.interval(confidence_interval, sample_size - 1)[1])
        return (
            sample_mean - t_positive * sample_standard_deviation / math.sqrt(sample_size),
            sample_mean + t_positive * sample_standard_deviation / math.sqrt(sample_size)
        )

    @staticmethod
    def mean_difference(sample_mean1: float, sample_size1: int, sample_mean2: float, sample_size2: int, sample_variance: float, confidence_interval: float) -> tuple[float, float]:
        """Intervalo de confiança para estimar a diferença de médias de duas populações"""
        degrees_of_freedom = sample_size1 + sample_size2 - 2
        combined_variance = (
            (sample_size1 - 1) * sample_variance + (sample_size2 - 1) * sample_variance
        ) / degrees_of_freedom
        t_positive = float(t.interval(confidence_interval, degrees_of_freedom)[1])
        return (
            sample_mean1 - sample_mean2 - t_positive * math.sqrt(combined_variance / sample_size1 + combined_variance / sample_size2),
            sample_mean1 - sample_mean2 + t_positive * math.sqrt(combined_variance / sample_size1 + combined_variance / sample_size2)
        )

    @staticmethod
    def variance(sample_variance: float, sample_size: int, confidence_interval: float) -> tuple[float, float]:
        """Intervalo de confiança para estimar a variância de uma população""" 
        degrees_of_freedom = sample_size - 1
        chi2_negative = float(chi2.interval(confidence_interval, degrees_of_freedom)[0])
        chi2_positive = float(chi2.interval(confidence_interval, degrees_of_freedom)[1])
        return (
            (degrees_of_freedom * sample_variance) / chi2_positive,
            (degrees_of_freedom * sample_variance) / chi2_negative
        )

    @staticmethod
    def proportion(sample_proportion: float, sample_size: int, confidence_interval: float) -> tuple[float, float]:
        """Intervalo de confiança para estimar a proporção de uma população"""
        if (sample_size * sample_proportion > 5 and sample_size * (1 - sample_proportion) > 5):
            z = float(norm.interval(confidence_interval)[1])
            return (
                sample_proportion - z * math.sqrt(sample_proportion * (1 - sample_proportion) / sample_size),
                sample_proportion + z * math.sqrt(sample_proportion * (1 - sample_proportion) / sample_size)
            )
        else:
            raise ValueError("n não é grande o suficiente para estimar proporção")

    @staticmethod
    def sample_size_mean_known(standard_deviation: float, confidence_interval: float, pilot_sample_size: int) -> float:
        """Estima o tamanho de amostra necessário para estimar a média, quando a média populacional é conhecida"""
        z = float(norm.interval(confidence_interval)[1])
        error = z * standard_deviation / math.sqrt(pilot_sample_size)
        return (z * standard_deviation / error) ** 2
    
    @staticmethod
    def sample_size_mean_unknown(sample_standard_deviation: float, confidence_interval: float, pilot_sample_size: int, error: float) -> float:
        """Estima o tamanho de amostra necessário para estimar a média, quando a média populacional é desconhecida"""
        degrees_of_freedom = pilot_sample_size - 1
        t_positive = float(t.interval(confidence_interval, degrees_of_freedom)[1])
        return (t_positive * sample_standard_deviation / error) ** 2

    @staticmethod
    def sample_size_proportion(sample_proportion: float, confidence_interval: float, error: float) -> float:
        """Estima o tamanho de amostra necessário para estimar a proporção de uma população"""
        z = float(norm.interval(confidence_interval)[1])
        return (z ** 2 * (sample_proportion * (1 - sample_proportion)) / error ** 2)

    @staticmethod
    def precision_mean_known(standard_deviation: float, confidence_interval: float, sample_size: int) -> float:
        """Estima a precisão da média, quando a média populacional é conhecida"""
        z = float(norm.interval(confidence_interval)[1])
        return z * standard_deviation / math.sqrt(sample_size)

    @staticmethod
    def precision_mean_unknown(sample_standard_deviation: float, confidence_interval: float, sample_size: int) -> float:
        """Estima a precisão da média, quando a média populacional é desconhecida"""
        degrees_of_freedom = sample_size - 1
        t_positive = float(t.interval(confidence_interval, degrees_of_freedom)[1])
        return t_positive * sample_standard_deviation / math.sqrt(sample_size)

    @staticmethod
    def precision_proportion(sample_proportion: float, sample_size: int, confidence_interval: float) -> float:
        """Estima a precisão da proporção de uma população"""
        z = float(norm.interval(confidence_interval)[1])
        return z * math.sqrt(sample_proportion * (1 - sample_proportion) / sample_size)


# === Questões ===
# 8) a)
result = ConfidenceInterval.mean_unknown(330, 19, math.sqrt(81), 0.95)
print(round(result[1], 1))

# 8) b)
result = ConfidenceInterval.mean_unknown(330, 19, math.sqrt(81), 0.90)
amplitude = result[1] - result[0]
print(round(amplitude, 1))

# 8) c)
result = ConfidenceInterval.sample_size_mean_unknown(math.sqrt(81), 0.90, 19, 3.5)
print("%.0f" % (round(result, 0)))