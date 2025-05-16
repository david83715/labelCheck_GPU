#pragma once

/** @file
 * Collection of helper and utility functions for Darknet.
 */


#include "darknet_internal.hpp"


namespace Darknet
{
	/// Convert to lowercase and remove all but alphanumerics.
	std::string convert_to_lowercase_alphanum(const std::string & arg);

	/** Convert the given text to plain alphanumeric ASCII string.  Remove whitespace, keep just alphanumeric and underscore.
	 * Good to use as a base for a filename.
	 */
	std::string text_to_simple_label(std::string txt);

	/// @{ Trim leading and trailing whitespace from the given string.
	std::string trim(const std::string & str);
	std::string & trim(std::string & str);
	/// @}

	/// @{ Simple conversion of the string to lowercase.
	std::string lowercase(const std::string & str);
	std::string & lowercase(std::string & str);
	/// @}

	/// Setup the new C++ charts.  This is called once just prior to starting training.  @see @ref Chart
	void initialize_new_charts(const Darknet::Network & net);

	/// Update the new C++ charts with the given loss and mAP% accuracy value.  This is called at every iteration.  @see @ref Chart
	void update_loss_in_new_charts(const int current_iteration, const float loss, const float seconds_remaining, const bool dont_show);

	void update_accuracy_in_new_charts(const int class_index, const float accuracy);

	std::string get_command_output(const std::string & cmd);

	void cfg_layers();
}
